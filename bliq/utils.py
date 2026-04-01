import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------
# METRICS
# ---------------------------
def compute_metrics(true, pred):
    """
    Compute R² and RMSE.
    """
    true = true.detach().cpu().numpy().reshape(-1)
    pred = pred.detach().cpu().numpy().reshape(-1)

    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    return r2, rmse


# ---------------------------
# FORWARD PLOT
# ---------------------------
def plot_forward(y_true, y_pred, title="Forward Mapping"):
    plt.figure(figsize=(5, 4))
    plt.scatter(y_true.cpu(), y_pred.cpu(), alpha=0.4)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------
# INVERSE COMPARISON
# ---------------------------
def plot_inverse(x_true, x_rec, title="Inverse Reconstruction"):
    plt.figure(figsize=(5, 4))
    plt.scatter(x_true[:, 0].cpu(), x_true[:, 1].cpu(), alpha=0.2, label="True")
    plt.scatter(x_rec[:, 0].cpu(), x_rec[:, 1].cpu(), alpha=0.2, label="Reconstructed")
    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ---------------------------
# MANIFOLD PLOT
# ---------------------------
def plot_manifold(x_rec, y, title="Learned Inverse Manifold"):
    plt.figure(figsize=(5, 4))
    plt.scatter(
        x_rec[:, 0].cpu(),
        x_rec[:, 1].cpu(),
        c=y.cpu(),
        cmap="viridis",
        s=10,
    )
    plt.colorbar(label="y value")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ---------------------------
# CONSISTENCY RESIDUAL
# ---------------------------
def plot_consistency(model, x_rec, y):
    with torch.no_grad():
        y_from_xrec = model.forward(x_rec)
        residual = torch.abs(y - y_from_xrec)

    plt.figure(figsize=(5, 4))
    plt.scatter(
        x_rec[:, 0].cpu(),
        x_rec[:, 1].cpu(),
        c=residual.cpu(),
        cmap="viridis",
        s=10,
    )
    plt.colorbar(label="|y - f(x_rec)|")
    plt.title("Consistency Residual")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def inverse_consistency_error(model, x_rec, y):
    """
    Measures validity of inverse solutions.

    Computes:
        ||y - f(x_rec)||

    Lower is better.
    """
    with torch.no_grad():
        y_pred = model.forward(x_rec)
        error = torch.abs(y - y_pred)

    return error.mean().item(), error.max().item()