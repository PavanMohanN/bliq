import torch
import numpy as np
import random

from bliq.model import BLiqNet
from bliq.utils import (
    compute_metrics,
    plot_forward,
    plot_inverse,
    plot_manifold,
    plot_consistency,
)

# ---------------------------
# SEED
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------
# DATA
# ---------------------------
def generate_data(n=2000):
    x = np.random.uniform(-2, 2, (n, 2))

    y = (
        np.sin(x[:, 0]) + 0.5 * x[:, 1]
    ).reshape(-1, 1)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


# ---------------------------
# BASELINE MLP
# ---------------------------
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.forward_net(x)

    def inverse(self, y):
        return self.inverse_net(y)


def train_mlp(model, x, y, epochs=800):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        opt.zero_grad()

        y_pred = model(x)
        x_rec = model.inverse(y)

        loss = ((y - y_pred)**2).mean() + ((x - x_rec)**2).mean()
        loss.backward()
        opt.step()

    return model


# ---------------------------
# MAIN
# ---------------------------
def main():

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    x, y = generate_data()
    x, y = x.to(device), y.to(device)

    # Train MLP
    print("\nTraining MLP...")
    mlp = train_mlp(MLP().to(device), x, y)

    # Train BLiqNet
    print("\nTraining BLiqNet...")
    bliq = BLiqNet(input_dim=2, output_dim=1).to(device)
    bliq.fit(x, y, epochs=800)

    # Evaluation
    with torch.no_grad():
        y_pred_mlp = mlp(x)
        x_rec_mlp = mlp.inverse(y)

        y_pred_bliq = bliq.forward(x)
        x_rec_bliq = bliq.inverse(y)

    # Metrics
    r2_mlp, rmse_mlp = compute_metrics(y, y_pred_mlp)
    r2_bliq, rmse_bliq = compute_metrics(y, y_pred_bliq)

    print("\n=== Forward Metrics ===")
    print(f"MLP     : R2={r2_mlp:.4f}, RMSE={rmse_mlp:.4f}")
    print(f"BLiqNet : R2={r2_bliq:.4f}, RMSE={rmse_bliq:.4f}")

    # Consistency
    y_from_xrec = bliq.forward(x_rec_bliq)
    consistency = torch.abs(y - y_from_xrec)

    print("\n=== Inverse Consistency ===")
    print(f"Mean Error: {consistency.mean().item():.4f}")
    print(f"Max Error : {consistency.max().item():.4f}")

    # Plots
    plot_forward(y, y_pred_mlp, "MLP Forward")
    plot_forward(y, y_pred_bliq, "BLiqNet Forward")

    plot_inverse(x, x_rec_mlp, "MLP Inverse (Collapse)")
    plot_inverse(x, x_rec_bliq, "BLiqNet Inverse (Structured)")

    plot_manifold(x_rec_bliq, y)
    plot_consistency(bliq, x_rec_bliq, y)


if __name__ == "__main__":
    main()