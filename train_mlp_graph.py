
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import MLP
from data_graph import data_train, data_test


# ============================================================
# Configuration
# ============================================================

SEED = 42
EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_SIZES = (64, 128, 128, 64, 32)

CHECKPOINT_DIR = "checkpoints/mlp_graph_baseline"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Validate datasets
# ============================================================

if not data_train or not data_test:
    raise RuntimeError("Training or testing dataset is empty.")

print("Training circuits:", list(data_train.keys()))
print("Held-out test circuits:", list(data_test.keys()))

for split_name, dataset in [("TRAIN", data_train), ("TEST", data_test)]:
    for name, (g, ts) in dataset.items():
        x = g.ndata["nf"].float()
        y = g.ndata["n_atslew"].float()

        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError(f"{name}: expected 10 input features, got {tuple(x.shape)}")

        if y.ndim != 2 or y.shape[1] != 8:
            raise ValueError(f"{name}: expected 8 targets, got {tuple(y.shape)}")

        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"{name}: NaN or infinite values detected.")

        print(
            f"{split_name:5s} {name:18s} "
            f"nodes={g.num_nodes():7d}, X={tuple(x.shape)}, Y={tuple(y.shape)}"
        )


# ============================================================
# Fit normalization statistics on training circuits only
# Each circuit contributes equally to the statistics.
# ============================================================

x_means = []
x_vars = []
y_means = []
y_vars = []

for name, (g, ts) in data_train.items():
    x = g.ndata["nf"].float()
    y = g.ndata["n_atslew"].float()

    x_means.append(x.mean(dim=0))
    x_vars.append(x.var(dim=0, unbiased=False))
    y_means.append(y.mean(dim=0))
    y_vars.append(y.var(dim=0, unbiased=False))

x_mean = torch.stack(x_means).mean(dim=0)
y_mean = torch.stack(y_means).mean(dim=0)

# Combine within-circuit variance and differences between circuit means.
x_var = torch.stack([
    v + (m - x_mean).pow(2)
    for m, v in zip(x_means, x_vars)
]).mean(dim=0)

y_var = torch.stack([
    v + (m - y_mean).pow(2)
    for m, v in zip(y_means, y_vars)
]).mean(dim=0)

x_std = x_var.sqrt().clamp_min(1e-6)
y_std = y_var.sqrt().clamp_min(1e-6)


def get_xy(dataset, name):
    g, ts = dataset[name]
    x = g.ndata["nf"].float()
    y = g.ndata["n_atslew"].float()

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    return x_scaled.to(DEVICE), y_scaled.to(DEVICE)


# ============================================================
# Model
# ============================================================

model = MLP(10, *HIDDEN_SIZES, 8).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_train_loss = float("inf")


# ============================================================
# Training
# ============================================================

print("\nStarting graph-node MLP training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0

    # Equal weighting per training circuit.
    for name in data_train:
        x, y = get_xy(data_train, name)
        pred = model(x)
        loss = F.mse_loss(pred, y) / len(data_train)
        loss.backward()
        total_loss += loss.item()

    optimizer.step()

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Mean normalized train MSE: {total_loss:.6f}"
        )

    if total_loss < best_train_loss:
        best_train_loss = total_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "hidden_sizes": HIDDEN_SIZES,
                "seed": SEED,
                "epoch": epoch,
                "train_loss": total_loss,
            },
            CHECKPOINT_PATH,
        )


print(f"\nBest checkpoint saved to: {CHECKPOINT_PATH}")


# ============================================================
# Evaluation on held-out circuits
# Predictions and targets are converted back to original units.
# ============================================================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_true = []
all_pred = []

print("\n========== HELD-OUT CIRCUIT METRICS ==========")
print("Metrics are computed in the original target units.")

with torch.no_grad():
    for name in data_test:
        x, y_scaled = get_xy(data_test, name)

        pred_scaled = model(x)

        y_true = (y_scaled * y_std + y_mean).cpu().numpy()
        y_pred = (pred_scaled * y_std + y_mean).cpu().numpy()

        all_true.append(y_true)
        all_pred.append(y_pred)

        print(f"\nCircuit: {name}")
        for channel in range(8):
            yt = y_true[:, channel]
            yp = y_pred[:, channel]

            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))

            # R² is undefined for a single observation.
            r2 = r2_score(yt, yp) if len(yt) > 1 else float("nan")

            print(
                f"  Channel {channel}: "
                f"MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.4f}"
            )

    # Pooled results weight every node equally.
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    print("\n========== POOLED HELD-OUT METRICS ==========")

    for channel in range(8):
        yt = y_true_all[:, channel]
        yp = y_pred_all[:, channel]

        print(
            f"Channel {channel}: "
            f"MAE={mean_absolute_error(yt, yp):.6f}, "
            f"RMSE={np.sqrt(mean_squared_error(yt, yp)):.6f}, "
            f"R2={r2_score(yt, yp):.4f}"
        )

    print("\n========== OVERALL POOLED METRICS ==========")
    yt = y_true_all.reshape(-1)
    yp = y_pred_all.reshape(-1)

    print(f"MAE : {mean_absolute_error(yt, yp):.6f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yt, yp)):.6f}")
    print(f"R2  : {r2_score(yt, yp):.4f}")

print("\nMLP baseline finished.")
