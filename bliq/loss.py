import torch

def bliq_loss(
    model,
    x,
    y,
    lambda_rec: float = 0.1,
    lambda_inv: float = 1.0,     # Default increased for better manifold mapping
    lambda_latent: float = 0.1,
    lambda_reg: float = 1e-4,
    noise_level: float = 0.5    # Noise injected during training for diversity
):
    """
    Computes BLiqNet dual-consistency loss with stochastic inverse support.

    Components:
        1. Forward loss:        y ≈ f(x)
        2. Reconstruction:      x ≈ g(y) (Supervised)
        3. Inverse feasibility: y ≈ f(g(y)) (Consistency - CRITICAL)
        4. Latent consistency:  h_f ≈ h_b
    """

    # 1. Forward pass (Standard)
    y_pred, h_f = model.forward_with_latent(x)
    loss_f = torch.mean((y - y_pred) ** 2)

    # 2. Inverse pass (Stochastic)
    # We inject small noise here so the ODE learns to map y to a valid 
    # neighborhood of x, rather than a single rigid coordinate.
    x_rec, h_b = model.inverse_with_latent(y, noise_level=noise_level)

    # 3. Reconstruction loss
    # How close is the reconstructed x to the original x?
    loss_rec = torch.mean((x - x_rec) ** 2)

    # 4. Inverse feasibility (Cycle Consistency)
    # If we pass our reconstructed x back through the forward model, 
    # do we get the original y? This is vital for many-to-one problems.
    y_from_xrec = model.forward(x_rec)
    loss_inv = torch.mean((y - y_from_xrec) ** 2)

    # 5. Latent consistency
    # Ensures the latent dynamics (Liquid ODE) are synchronized 
    # across both directions.
    loss_latent = torch.mean((h_f - h_b) ** 2)

    # 6. Weight Regularization (L2)
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