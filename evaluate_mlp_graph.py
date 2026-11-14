
import os
import numpy as np
import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import MLP
from data_graph import data_test

CHECKPOINT_PATH = os.path.join(
    "checkpoints", "mlp_graph_baseline", "best_model.pth"
)
DEVICE = torch.device("cpu")

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"MLP checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

hidden_sizes = tuple(checkpoint["hidden_sizes"])
model = MLP(10, *hidden_sizes, 8).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

x_mean = checkpoint["x_mean"].to(DEVICE)
x_std = checkpoint["x_std"].to(DEVICE)
y_mean = checkpoint["y_mean"].to(DEVICE)
y_std = checkpoint["y_std"].to(DEVICE)

all_true = []
all_pred = []

print("Loaded MLP checkpoint:", CHECKPOINT_PATH)
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Held-out circuits:", list(data_test.keys()))

with torch.no_grad():
    for name, (g, ts) in data_test.items():
        x = g.ndata["nf"].float().to(DEVICE)
        y = g.ndata["n_atslew"].float().to(DEVICE)

        x_scaled = (x - x_mean) / x_std
        pred_scaled = model(x_scaled)

        y_true = y.cpu().numpy()
        y_pred = (pred_scaled * y_std + y_mean).cpu().numpy()

        all_true.append(y_true)
        all_pred.append(y_pred)

        print(f"\n--- {name} ---")

        for channel in range(8):
            yt = y_true[:, channel]
            yp = y_pred[:, channel]

            print(
                f"Channel {channel} "
                f"({'Arrival Time' if channel < 4 else 'Transformed Slew'}): "
                f"R2={r2_score(yt, yp):.5f}, "
                f"MAE={mean_absolute_error(yt, yp):.6f}, "
                f"RMSE={np.sqrt(mean_squared_error(yt, yp)):.6f}"
            )

        for label, channels in (
            ("Arrival Time (ch 0-3)", range(0, 4)),
            ("Transformed Slew (ch 4-7)", range(4, 8)),
        ):
            idx = list(channels)
            yt = y_true[:, idx].reshape(-1)
            yp = y_pred[:, idx].reshape(-1)

            print(
                f"{label}: "
                f"R2={r2_score(yt, yp):.5f}, "
                f"MAE={mean_absolute_error(yt, yp):.6f}, "
                f"RMSE={np.sqrt(mean_squared_error(yt, yp)):.6f}"
            )

y_true_all = np.concatenate(all_true, axis=0)
y_pred_all = np.concatenate(all_pred, axis=0)

print("\n======= POOLED HELD-OUT METRICS =======")

for channel in range(8):
    yt = y_true_all[:, channel]
    yp = y_pred_all[:, channel]

    print(
        f"Channel {channel}: "
        f"R2={r2_score(yt, yp):.5f}, "
        f"MAE={mean_absolute_error(yt, yp):.6f}, "
        f"RMSE={np.sqrt(mean_squared_error(yt, yp)):.6f}"
    )

for label, channels in (
    ("Arrival Time (ch 0-3)", range(0, 4)),
    ("Transformed Slew (ch 4-7)", range(4, 8)),
):
    idx = list(channels)
    yt = y_true_all[:, idx].reshape(-1)
    yp = y_pred_all[:, idx].reshape(-1)

    print(
        f"{label}: "
        f"R2={r2_score(yt, yp):.5f}, "
        f"MAE={mean_absolute_error(yt, yp):.6f}, "
        f"RMSE={np.sqrt(mean_squared_error(yt, yp)):.6f}"
    )

print("\nMLP evaluation complete. No model was retrained.")
