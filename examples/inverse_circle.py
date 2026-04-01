import torch
import numpy as np
import random
import torch.nn as nn

# Ensure your model file is named 'model_circle.py' or update this import
from bliq.model_circle import BLiqNet 
from bliq.utils import (
    compute_metrics,
    plot_forward,
    plot_inverse,
    plot_manifold,
    plot_consistency,
    plot_liquid_constants
)

# ---------------------------
# REPRODUCIBILITY
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# DATA GENERATION
# ---------------------------
def generate_data(n=2500):
    # Expanded range to ensure full circle coverage
    x = np.random.uniform(-2, 2, (n, 2))
    # y = x1^2 + x2^2
    y = (x[:, 0]**2 + x[:, 1]**2).reshape(-1, 1)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

# ---------------------------
# BASELINE MLP
# ---------------------------
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

def train_mlp(model, x, y, epochs=1000):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
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

    # 1. Data
    x, y = generate_data()
    x, y = x.to(device), y.to(device)

    # 2. Train MLP (Baseline)
    print("\nTraining MLP...")
    mlp = train_mlp(MLP().to(device), x, y)

    # 3. Train BLiqNet
    print("\nTraining BLiqNet (Symmetry-Aware)...")
    # Using hidden_dim=128 and t_span=(0, 2) as discussed for higher capacity
    bliq = BLiqNet(input_dim=2, output_dim=1, hidden_dim=128, t_span=(0.0, 2.0)).to(device)
    
    # Training with manifold-optimized lambdas
    bliq.fit(
        x, y, 
        epochs=1500, 
        lambda_rec=0.01,   # Low: Don't force specific points
        lambda_inv=2.0,    # High: Force the circle physics
        lambda_latent=0.1
    )

    # 4. Evaluation
    print("\nEvaluating Manifold Sampling...")
    with torch.no_grad():
        y_pred_mlp = mlp(x)
        y_pred_bliq = bliq.forward(x)
        x_rec_mlp = mlp.inverse(y)

        # Sampling many points (20) for each y with high noise_level (0.8)
        # to trigger the Angular Symmetry Injection and fill the circle.
        num_samples = 20
        x_rec_samples = []
        for _ in range(num_samples):
            sample = bliq.inverse(y, noise_level=0.8)
            x_rec_samples.append(sample)
        
        x_rec_bliq_multi = torch.cat(x_rec_samples, dim=0)
        
        # Clean/Deterministic pass for consistency metrics
        x_rec_bliq_clean = bliq.inverse(y, noise_level=0.0)

    # 5. Metrics
    r2_mlp, rmse_mlp = compute_metrics(y, y_pred_mlp)
    r2_bliq, rmse_bliq = compute_metrics(y, y_pred_bliq)

    print("\n=== Forward Metrics ===")
    print(f"MLP     : R2={r2_mlp:.4f}, RMSE={rmse_mlp:.4f}")
    print(f"BLiqNet : R2={r2_bliq:.4f}, RMSE={rmse_bliq:.4f}")

    # 6. Consistency
    y_from_xrec = bliq.forward(x_rec_bliq_clean)
    consistency_error = torch.abs(y - y_from_xrec)

    print("\n=== Inverse Consistency (Clean) ===")
    print(f"Mean Error: {consistency_error.mean().item():.4f}")
    print(f"Max Error : {consistency_error.max().item():.4f}")

    # 7. Visualizations
    plot_forward(y, y_pred_mlp, title="MLP Forward (y = x²)")
    plot_forward(y, y_pred_bliq, title="BLiqNet Forward (y = x²)")

    plot_inverse(x, x_rec_mlp, title="MLP Inverse (Mode Collapse)")
    
    # Plotting multi-sampled reconstruction to see the full 360 ring
    x_true_multi = x.repeat(num_samples, 1)
    plot_inverse(x_true_multi, x_rec_bliq_multi, title="BLiqNet Inverse (Full Manifold Reconstruction)")

    plot_manifold(x_rec_bliq_clean, y, title="BLiqNet Learned Manifold Spine")
    plot_consistency(bliq, x_rec_bliq_clean, y)
    plot_liquid_constants(bliq)

if __name__ == "__main__":
    main()