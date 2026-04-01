import torch
import torch.optim as optim
from .loss import bliq_loss


def train_bliq(
    model,
    x,
    y,
    epochs: int = 500,
    lr: float = 1e-3,
    lambda_rec: float = 0.1,
    lambda_inv: float = 0.1,
    lambda_latent: float = 0.1,
    verbose: bool = True,
):
    """
    Training loop for BLiqNet.

    Args:
        model: BLiqNet instance
        x: input tensor
        y: output tensor
        epochs: number of training iterations
        lr: learning rate
        lambda_*: loss weights
        verbose: print progress

    Returns:
        trained model
    """

    device = next(model.parameters()).device

    x = x.to(device)
    y = y.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, components = bliq_loss(
            model,
            x,
            y,
            lambda_rec=lambda_rec,
            lambda_inv=lambda_inv,
            lambda_latent=lambda_latent,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(
                f"[Epoch {epoch}] "
                f"Loss: {loss.item():.4f} | "
                f"F: {components['forward']:.4f} "
                f"Inv: {components['inverse']:.4f} "
                f"Lat: {components['latent']:.4f}"
            )

    return model