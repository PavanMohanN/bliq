import torch
import torch.nn as nn


class LiquidODEFunc(nn.Module):
    """
    Defines the continuous-time dynamics:
        dh/dt = f(h, u)

    where:
        h = latent state
        u = input (projected)

    This class avoids lambda closures by storing input explicitly.
    """

    def __init__(self, dim_h: int):
        super().__init__()

        self.dim_h = dim_h

        # Hidden-to-hidden dynamics
        self.linear_h = nn.Linear(dim_h, dim_h)

        # Input injection (same dimension)
        self.linear_u = nn.Linear(dim_h, dim_h)

        self.activation = torch.tanh

        # Placeholder for input (set during forward pass)
        self.u = None

    def set_input(self, u: torch.Tensor):
        """
        Store input for ODE integration.
        Shape: (batch_size, dim_h)
        """
        self.u = u

    def forward(self, t, h):
        """
        ODE function evaluation.

        Args:
            t: time (unused but required)
            h: current latent state (batch, dim_h)

        Returns:
            dh/dt
        """
        if self.u is None:
            raise RuntimeError("Input u not set. Call set_input(u) before integration.")

        return self.activation(
            self.linear_h(h) + self.linear_u(self.u)
        )