import os
import csv
import numpy as np
import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import MLP, TimingGCN
from data_graph import data_test

DEVICE = torch.device("cpu")

GNN_PATH = r"checkpoints\experiment_6train4test_no_gt\99.pth"
MLP_PATH = r"checkpoints\mlp_graph_baseline\best_model.pth"
OUT_DIR = r"checkpoints\paper_figures"
CSV_PATH = os.path.join(OUT_DIR, "endpoint_metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)

for path in (GNN_PATH, MLP_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

# Load GNN
gnn = TimingGCN().to(DEVICE)
gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
gnn.eval()

# Load MLP
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
    model_name: {group: {"true": [], "pred": []}
                 for group in channel_groups}
    for model_name in ("GNN", "MLP")
}

def add_metrics(model_name, circuit, group, y_true, y_pred, n_endpoints):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(f"No target values for {circuit}, {group}")

    row = {
        "model": model_name,
        "circuit": circuit,
        "target": group,
        "endpoint_count": n_endpoints,
        "target_values": len(y_true),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    rows.append(row)

    print(
        f"{model_name:4} | {circuit:16} | {group:17} | "
        f"n={n_endpoints:5d} | "
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

        truth = g.ndata["n_atslew"].float().to(DEVICE)
        nf = g.ndata["nf"].float().to(DEVICE)

        # GNN predicts all graph nodes; evaluate only timing endpoints.
        pred_gnn = gnn(g, ts, groundtruth=False)[2]

        # MLP predicts all graph nodes; restore original target scale.
        nf_scaled = (nf - x_mean) / x_std
        pred_mlp = mlp(nf_scaled) * y_std + y_mean

        endpoint_truth = truth[endpoints].cpu().numpy()
        endpoint_gnn = pred_gnn[endpoints].cpu().numpy()
        endpoint_mlp = pred_mlp[endpoints].cpu().numpy()

        for model_name, prediction in (
            ("GNN", endpoint_gnn),
            ("MLP", endpoint_mlp),
        ):
            for group, channels in channel_groups.items():
                yt = endpoint_truth[:, channels].reshape(-1)
                yp = prediction[:, channels].reshape(-1)

                add_metrics(
                    model_name, circuit, group, yt, yp, len(endpoints)
                )

                pooled[model_name][group]["true"].append(yt)
                pooled[model_name][group]["pred"].append(yp)

print("\n========== POOLED ENDPOINT METRICS ==========")

for model_name in ("GNN", "MLP"):
    for group in channel_groups:
        yt = np.concatenate(pooled[model_name][group]["true"])
        yp = np.concatenate(pooled[model_name][group]["pred"])

        add_metrics(
            model_name, "ALL_HELD_OUT", group, yt, yp,
            sum(ts["endpoints"].numel() for _, ts in data_test.values())
        )

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model", "circuit", "target", "endpoint_count",
        "target_values", "R2", "MAE", "RMSE"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved endpoint metrics to: {CSV_PATH}")
print("No models were retrained or checkpoints modified.")
