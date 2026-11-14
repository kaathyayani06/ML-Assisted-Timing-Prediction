import argparse
import os
import random
import time
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import tee

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from data_graph import data_train, data_test
from model import TimingGCN


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

random.seed(8026728)
np.random.seed(8026728)
torch.manual_seed(8026728)


parser = argparse.ArgumentParser(
    description="Train and evaluate the timing-prediction GNN."
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="Number of training epochs. Use --epochs 2 for a smoke test.",
)

parser.add_argument(
    "--test_iter",
    type=int,
    default=None,
    help="Load and evaluate a saved epoch checkpoint instead of training.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint directory name under the project's checkpoints folder.",
)

parser.set_defaults(
    netdelay=True,
    celldelay=True,
    groundtruth=True,
)

parser.add_argument(
    "--no_netdelay",
    dest="netdelay",
    action="store_false",
    help="Disable net-delay training supervision.",
)

parser.add_argument(
    "--no_celldelay",
    dest="celldelay",
    action="store_false",
    help="Disable cell-delay training supervision.",
)

parser.add_argument(
    "--no_groundtruth",
    dest="groundtruth",
    action="store_false",
    help="Disable ground-truth breakdown during training.",
)


# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------

model = TimingGCN()


# ------------------------------------------------------------
# 3. R2 helper
#    sklearn convention: actual values first, predictions second
# ------------------------------------------------------------

def calculate_r2(truth, pred):
    truth = truth.detach().cpu().numpy().reshape(-1)
    pred = pred.detach().cpu().numpy().reshape(-1)

    return r2_score(truth, pred)


# ------------------------------------------------------------
# 4. Evaluate timing predictions
# ------------------------------------------------------------

def test(model, save_dump=True):
    model.eval()

    with torch.no_grad():
        print("\n======= Training dataset ======")

        for k, (g, ts) in data_train.items():
            start_time = time.time()

            # Model returns:
            # predicted net delays, cell delays, arrival-time/slew values
            pred = model(g, ts, groundtruth=False)[2]
            truth = g.ndata["n_atslew"]

            r2 = calculate_r2(truth[:, :4], pred[:, :4])

            print(
                f"{k:15} R2 = {r2:.5f}, "
                f"time = {time.time() - start_time:.2f}s"
            )

        print("\n======= Held-out test dataset ======")

        for k, (g, ts) in data_test.items():
            start_time = time.time()

            pred = model(g, ts, groundtruth=False)[2]
            truth = g.ndata["n_atslew"]

            r2 = calculate_r2(truth[:, :4], pred[:, :4])

            print(
                f"{k:15} R2 = {r2:.5f}, "
                f"time = {time.time() - start_time:.2f}s"
            )

            # Preserve the existing NPZ field names for compatibility.
            # These fields are columns 0 and 1 of n_atslew; verify their
            # exact semantics before interpreting them as slack values.
            if save_dump and k == "cic_decimator":
                output_dir = CHECKPOINT_DIR / "18_slacksdump"
                output_dir.mkdir(parents=True, exist_ok=True)

                np.savez(
                    output_dir / "cic_decimator_gnn.npz",
                    se=pred[:, 0].detach().cpu().numpy(),
                    sl=pred[:, 1].detach().cpu().numpy(),
                    se_truth=truth[:, 0].detach().cpu().numpy(),
                    sl_truth=truth[:, 1].detach().cpu().numpy(),
                )

                print(
                    "Saved cic_decimator_gnn.npz "
                    "(existing field names preserved)."
                )


# ------------------------------------------------------------
# 5. Evaluate net-delay predictions
# ------------------------------------------------------------

def test_netdelay(model):
    model.eval()

    with torch.no_grad():
        for split_name, dataset in (
            ("Training dataset", data_train),
            ("Held-out test dataset", data_test),
        ):
            print(f"\n======= {split_name}: net delays ======")

            for k, (g, ts) in dataset.items():
                pred = model(g, ts, groundtruth=False)[0]
                truth = g.ndata["n_net_delays_log"]

                r2 = calculate_r2(truth, pred)

                print(f"{k:15} R2 = {r2:.5f}")


# ------------------------------------------------------------
# 6. Final detailed metrics
# ------------------------------------------------------------


def evaluate_metrics(model):
    model.eval()

    print("\n======= Final GNN Test Metrics ======")

    channel_groups = {
        "Arrival Time (ch 0-3)": range(0, 4),
        "Transformed Slew (ch 4-7)": range(4, 8),
    }

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for k, (g, ts) in data_test.items():
            pred = model(g, ts, groundtruth=False)[2]
            truth = g.ndata["n_atslew"]

            pred_np = pred.detach().cpu().numpy()
            truth_np = truth.detach().cpu().numpy()

            all_predictions.append(pred_np)
            all_targets.append(truth_np)

            print(f"\n--- {k} ---")

            # Metrics for each of the eight channels
            for i in range(8):
                y_pred = pred_np[:, i].reshape(-1)
                y_true = truth_np[:, i].reshape(-1)

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                if i < 4:
                    target_name = "Arrival Time"
                else:
                    target_name = "Transformed Slew"

                print(
                    f"Channel {i} ({target_name:17}) "
                    f"R2 = {r2:.5f}, "
                    f"MAE = {mae:.6f}, "
                    f"RMSE = {rmse:.6f}"
                )

            # Aggregate metrics within each target group
            for group_name, channels in channel_groups.items():
                y_pred = pred_np[:, list(channels)].reshape(-1)
                y_true = truth_np[:, list(channels)].reshape(-1)

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                print(
                    f"{group_name:29} "
                    f"R2 = {r2:.5f}, "
                    f"MAE = {mae:.6f}, "
                    f"RMSE = {rmse:.6f}"
                )

    # Pooled metrics across all held-out circuits
    pooled_pred = np.concatenate(all_predictions, axis=0)
    pooled_truth = np.concatenate(all_targets, axis=0)

    print("\n======= Pooled Held-out Metrics ======")

    for i in range(8):
        y_pred = pooled_pred[:, i]
        y_true = pooled_truth[:, i]

        print(
            f"Channel {i}: "
            f"R2 = {r2_score(y_true, y_pred):.5f}, "
            f"MAE = {mean_absolute_error(y_true, y_pred):.6f}, "
            f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}"
        )

    for group_name, channels in channel_groups.items():
        idx = list(channels)
        y_pred = pooled_pred[:, idx].reshape(-1)
        y_true = pooled_truth[:, idx].reshape(-1)

        print(
            f"{group_name:29} "
            f"R2 = {r2_score(y_true, y_pred):.5f}, "
            f"MAE = {mean_absolute_error(y_true, y_pred):.6f}, "
            f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}"
        )


# ------------------------------------------------------------
# 7. Training
# ------------------------------------------------------------

def train(model, args):
    if not data_train:
        raise RuntimeError("No training circuits were loaded.")

    if not data_test:
        raise RuntimeError("No held-out test circuits were loaded.")

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    batch_size = min(7, len(data_train))

    print("\n======= Training configuration ======")
    print(f"Training circuits : {list(data_train.keys())}")
    print(f"Test circuits     : {list(data_test.keys())}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch size        : {batch_size}")
    print(f"Net-delay loss    : {args.netdelay}")
    print(f"Cell-delay loss   : {args.celldelay}")
    print(f"Ground-truth mode : {args.groundtruth}")

    for epoch in range(args.epochs):
        model.train()

        train_net_total = 0.0
        train_cell_total = 0.0
        train_ats_total = 0.0

        optimizer.zero_grad()

        # Retain the original approach of accumulating gradients
        # across the sampled training circuits.
        selected_circuits = random.sample(
            list(data_train.items()),
            batch_size,
        )

        for k, (g, ts) in selected_circuits:
            (
                pred_net_delays,
                pred_cell_delays,
                pred_atslew,
            ) = model(
                g,
                ts,
                groundtruth=args.groundtruth,
            )

            # Net-delay supervision
            if args.netdelay:
                loss_net = F.mse_loss(
                    pred_net_delays,
                    g.ndata["n_net_delays_log"],
                )
            else:
                loss_net = pred_net_delays.sum() * 0.0

            # Cell-delay supervision
            if args.celldelay:
                loss_cell = F.mse_loss(
                    pred_cell_delays,
                    g.edges["cell_out"].data["e_cell_delays"],
                )
            else:
                loss_cell = pred_cell_delays.sum() * 0.0

            # Arrival-time/slew supervision
            loss_ats = F.mse_loss(
                pred_atslew,
                g.ndata["n_atslew"],
            )

            total_loss = loss_net + loss_cell + loss_ats
            total_loss.backward()

            train_net_total += loss_net.item()
            train_cell_total += loss_cell.item()
            train_ats_total += loss_ats.item()

            # Release references before processing the next circuit.
            del (
                pred_net_delays,
                pred_cell_delays,
                pred_atslew,
                loss_net,
                loss_cell,
                loss_ats,
                total_loss,
            )

        optimizer.step()

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"Net loss: {train_net_total / batch_size:.6f} | "
            f"Cell loss: {train_cell_total / batch_size:.6f} | "
            f"Arrival-time/slew loss: {train_ats_total / batch_size:.6f}"
        )

        # IMPORTANT:
        # Do not evaluate the held-out test set at every epoch.
        # This helps keep it separate from training-time model decisions.

    # --------------------------------------------------------
    # 8. Save final model
    # --------------------------------------------------------

    if args.checkpoint:
        save_dir = CHECKPOINT_DIR / args.checkpoint
        save_dir.mkdir(parents=True, exist_ok=True)

        # Preserve numeric checkpoint naming for --test_iter.
        checkpoint_path = save_dir / f"{args.epochs - 1}.pth"

        torch.save(model.state_dict(), checkpoint_path)

        print(f"\nSaved model to: {checkpoint_path}")

    # --------------------------------------------------------
    # 9. Evaluate the held-out test set once, after training
    # --------------------------------------------------------

    print("\nTraining complete. Running final evaluation.")

    test(model)
    evaluate_metrics(model)


# ------------------------------------------------------------
# 10. Main
# ------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_dir = CHECKPOINT_DIR / args.checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = checkpoint_dir / "stdout.log"
        stderr_path = checkpoint_dir / "stderr.log"

        if args.test_iter is not None:
            checkpoint_path = checkpoint_dir / f"{args.test_iter}.pth"

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location="cpu",
                )
            )

            print(f"Loaded model from: {checkpoint_path}")

            test(model)
            test_netdelay(model)
            evaluate_metrics(model)

        else:
            print(f"Saving logs and checkpoints to: {checkpoint_dir}")

            with tee.StdoutTee(str(stdout_path)), tee.StderrTee(
                str(stderr_path)
            ):
                train(model, args)

    else:
        if args.test_iter is not None:
            raise ValueError(
                "--test_iter requires --checkpoint."
            )

        print(
            "Warning: no checkpoint directory specified. "
            "The trained model and logs will not be saved."
        )

        train(model, args)