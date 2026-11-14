
import os
import csv
import shutil
from pathlib import Path

import dgl
import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from model import MLP, TimingGCN
from data_graph import data_test

DEVICE = torch.device("cpu")

GNN_PATH = r"checkpoints\experiment_6train4test_no_gt\99.pth"
MLP_PATH = r"checkpoints\mlp_graph_baseline\best_model.pth"

OUT_DIR = r"checkpoints\paper_figures"
CSV_PATH = os.path.join(OUT_DIR, "endpoint_metrics_audited.csv")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "8_rat"

os.makedirs(OUT_DIR, exist_ok=True)

for path in (GNN_PATH, MLP_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

# Load GNN checkpoint.
gnn = TimingGCN().to(DEVICE)
gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
gnn.eval()

# Load MLP checkpoint.
mlp_ckpt = torch.load(MLP_PATH, map_location=DEVICE)
hidden_sizes = tuple(mlp_ckpt["hidden_sizes"])

mlp = MLP(10, *hidden_sizes, 8).to(DEVICE)
mlp.load_state_dict(mlp_ckpt["model_state_dict"])
mlp.eval()

x_mean = mlp_ckpt["x_mean"].to(DEVICE)
x_std = mlp_ckpt["x_std"].to(DEVICE)
y_mean = mlp_ckpt["y_mean"].to(DEVICE)
y_std = mlp_ckpt["y_std"].to(DEVICE)

channel_groups = {
    "Arrival Time": list(range(0, 4)),
    "Transformed Slew": list(range(4, 8)),
}

rows = []
pooled = {
    model_name: {
        group: {"true": [], "pred": []}
        for group in channel_groups
    }
    for model_name in ("GNN", "MLP")
}


def add_metrics(
    model_name, circuit, group, y_true, y_pred,
    n_endpoints, n_valid_values
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            f"No valid target values for {circuit}, {group}"
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Truth/prediction length mismatch for {circuit}, {group}"
        )

    row = {
        "model": model_name,
        "circuit": circuit,
        "target": group,
        "endpoint_count": n_endpoints,
        "target_values": len(y_true),
        "valid_target_values": n_valid_values,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    rows.append(row)

    print(
        f"{model_name:4} | {circuit:16} | {group:17} | "
        f"endpoints={n_endpoints:5d} | "
        f"values={len(y_true):5d} | "
        f"R2={row['R2']: .5f} | "
        f"MAE={row['MAE']:.6f} | "
        f"RMSE={row['RMSE']:.6f}"
    )


with torch.no_grad():
    for circuit, (g, ts) in data_test.items():
        endpoints = ts["endpoints"].long()

        if endpoints.numel() == 0:
            print(f"Skipping {circuit}: no timing endpoints")
            continue

        # Load original graph to recover the preprocessed-away
        # invalid arrival-time information.
        graph_path = DATA_DIR / f"{circuit}.graph.bin"

        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Original graph not found: {graph_path}"
            )

        original_graphs, _ = dgl.load_graphs(str(graph_path))
        raw_graph = original_graphs[0]

        if raw_graph.num_nodes() != g.num_nodes():
            raise ValueError(
                f"Node count mismatch for {circuit}: "
                f"original={raw_graph.num_nodes()}, "
                f"processed={g.num_nodes()}"
            )

        raw_ats = raw_graph.ndata["n_ats"][endpoints].float()

        # Per-channel mask: invalid values were defined in
        # data_graph.py as abs(n_ats) > 1e20.
        valid_arrival = (
            torch.isfinite(raw_ats)
            & (torch.abs(raw_ats) <= 1e20)
        ).cpu().numpy()

        # Ground truth and model predictions.
        truth = g.ndata["n_atslew"].float().to(DEVICE)
        nf = g.ndata["nf"].float().to(DEVICE)

        pred_gnn = gnn(g, ts, groundtruth=False)[2]

        nf_scaled = (nf - x_mean) / x_std
        pred_mlp = mlp(nf_scaled) * y_std + y_mean

        endpoint_truth = truth[endpoints].cpu().numpy()
        endpoint_gnn = pred_gnn[endpoints].cpu().numpy()
        endpoint_mlp = pred_mlp[endpoints].cpu().numpy()

        # The transformed-slew targets are channels 4–7.
        # Require finite targets and finite predictions.
        # Do not reject zeros: zero raw slew can be valid.
        for model_name, prediction in (
            ("GNN", endpoint_gnn),
            ("MLP", endpoint_mlp),
        ):
            for group, channels in channel_groups.items():
                yt_matrix = endpoint_truth[:, channels]
                yp_matrix = prediction[:, channels]

                if group == "Arrival Time":
                    valid = (
                        valid_arrival
                        & np.isfinite(yt_matrix)
                        & np.isfinite(yp_matrix)
                    )
                else:
                    # Apply the corresponding original arrival-time
                    # validity mask to transformed-slew channels.
                    valid = (
                        valid_arrival
                        & np.isfinite(yt_matrix)
                        & np.isfinite(yp_matrix)
                    )
                    
                yt = yt_matrix[valid]
                yp = yp_matrix[valid]

                valid_count = int(valid.sum())

                add_metrics(
                    model_name=model_name,
                    circuit=circuit,
                    group=group,
                    y_true=yt,
                    y_pred=yp,
                    n_endpoints=len(endpoints),
                    n_valid_values=valid_count,
                )

                pooled[model_name][group]["true"].append(yt)
                pooled[model_name][group]["pred"].append(yp)

print("\n========== POOLED AUDITED ENDPOINT METRICS ==========")

for model_name in ("GNN", "MLP"):
    for group in channel_groups:
        yt = np.concatenate(pooled[model_name][group]["true"])
        yp = np.concatenate(pooled[model_name][group]["pred"])

        add_metrics(
            model_name=model_name,
            circuit="ALL_HELD_OUT",
            group=group,
            y_true=yt,
            y_pred=yp,
            n_endpoints=sum(
                ts["endpoints"].numel()
                for _, ts in data_test.values()
            ),
            n_valid_values=len(yt),
        )

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "circuit",
            "target",
            "endpoint_count",
            "target_values",
            "valid_target_values",
            "R2",
            "MAE",
            "RMSE",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved audited metrics to: {CSV_PATH}")
print("No models were retrained or checkpoints modified.")
