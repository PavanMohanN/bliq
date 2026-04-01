import torch


def bliq_loss(
    model,
    x,
    y,
    lambda_rec: float = 0.1,
    lambda_inv: float = 0.1,
    lambda_latent: float = 0.1,
    lambda_reg: float = 1e-4,
):
    """
    Computes BLiqNet dual-consistency loss.

    Components:
        1. Forward loss:        y ≈ f(x)
        2. Reconstruction:      x ≈ g(y)
        3. Inverse feasibility: y ≈ f(g(y))
        4. Latent consistency:  h_f ≈ h_b

    Args:
        model: BLiqNet instance
        x: input tensor (batch, input_dim)
        y: output tensor (batch, output_dim)

    Returns:
        total_loss, dict of individual components
    """

    # Forward pass
    y_pred, h_f = model.forward_with_latent(x)

    # Inverse pass
    x_rec, h_b = model.inverse_with_latent(y)

    # 1. Forward loss
    loss_f = torch.mean((y - y_pred) ** 2)

    # 2. Reconstruction loss
    loss_rec = torch.mean((x - x_rec) ** 2)

    # 3. Inverse feasibility
    y_from_xrec = model.forward(x_rec)
    loss_inv = torch.mean((y - y_from_xrec) ** 2)

    # 4. Latent consistency
    loss_latent = torch.mean((h_f - h_b) ** 2)

    # 5. Regularization
    loss_reg = 0.0
    for p in model.parameters():
        loss_reg += torch.sum(p ** 2)

    loss_reg = lambda_reg * loss_reg

    # Total loss
    total_loss = (
        loss_f
        + lambda_rec * loss_rec
        + lambda_inv * loss_inv
        + lambda_latent * loss_latent
        + loss_reg
    )

    return total_loss, {
        "forward": loss_f.item(),
        "reconstruction": loss_rec.item(),
        "inverse": loss_inv.item(),
        "latent": loss_latent.item(),
        "reg": loss_reg.item(),
    }