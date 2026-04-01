import torch

def bliq_loss(
    model,
    x,
    y,
    lambda_rec: float = 0.01,    # Lowered: Stop 'magnetizing' to specific points
    lambda_inv: float = 2.0,     # Increased: Prioritize ANY valid point on the circle
    lambda_latent: float = 0.1,
    lambda_reg: float = 1e-4,
    noise_level: float = 0.2     # Balanced noise for training diversity
):
    """
    Refactored BLiqNet Loss for Circle Manifolds.
    
    Novelty: 
        Shifts the objective from 'Exact Point Estimation' (Standard MLP) 
        to 'Manifold Feasibility' (Stochastic Liquid ODE).
    """

    # 1. Forward pass (Standard Supervised Learning)
    # y ≈ f(x)
    y_pred, h_f = model.forward_with_latent(x)
    loss_f = torch.mean((y - y_pred) ** 2)

    # 2. Inverse pass (Stochastic Manifold Sampling)
    # x_rec ≈ g(y, angle)
    # The noise_level here triggers the Angular Symmetry Injection in model_circle.py
    x_rec, h_b = model.inverse_with_latent(y, noise_level=noise_level)

    # 3. Reconstruction loss (Supervised Anchor)
    # Since y = x1^2 + x2^2 is many-to-one, we lower this weight (lambda_rec).
    # This prevents the 'arc' bias and allows the ODE to explore the full ring.
    loss_rec = torch.mean((x - x_rec) ** 2)

    # 4. Inverse feasibility (Cycle Consistency - THE CORE OF NOVELTY)
    # y ≈ f(g(y, angle))
    # This ensures that no matter WHICH angle the noise picks, the 
    # resulting x_rec is always at the correct radius.
    y_from_xrec = model.forward(x_rec)
    loss_inv = torch.mean((y - y_from_xrec) ** 2)

    # 5. Latent consistency
    # Ensures the Liquid ODE dynamics remain synchronized in both directions.
    loss_latent = torch.mean((h_f - h_b) ** 2)

    # 6. Weight Regularization
    loss_reg = 0.0
    for p in model.parameters():
        loss_reg += torch.sum(p ** 2)
    loss_reg = lambda_reg * loss_reg

    # Total loss weighted for Manifold Learning
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