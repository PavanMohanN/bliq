import torch
import torch.nn as nn
from torchdiffeq import odeint

from bliq.trainer import train_bliq

from .ode import LiquidODEFunc


class BLiqNet(nn.Module):
    """
    Bidirectional Liquid Neural Network

    Core idea:
        - Shared latent dynamics (ODE)
        - Forward head: x → y
        - Inverse head: y → x
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        t_span: tuple = (0.0, 1.0),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input projections
        self.input_x = nn.Linear(input_dim, hidden_dim)
        self.input_y = nn.Linear(output_dim, hidden_dim)

        # Shared ODE dynamics
        self.ode_func = LiquidODEFunc(hidden_dim)

        # Heads
        self.forward_head = nn.Linear(hidden_dim, output_dim)
        self.inverse_head = nn.Linear(hidden_dim, input_dim)

        # Time span
        self.register_buffer(
            "t",
            torch.tensor([t_span[0], t_span[1]], dtype=torch.float32)
        )

    # ---------------------------
    # INTERNAL: ODE evolution
    # ---------------------------
    def _evolve(self, u_proj: torch.Tensor):
        """
        Evolves latent state using ODE.

        Args:
            u_proj: (batch, hidden_dim)

        Returns:
            h_T: final latent state
        """
        self.ode_func.set_input(u_proj)

        h0 = u_proj  # empirically stable choice

        h_t = odeint(self.ode_func, h0, self.t)

        h_T = h_t[-1]
        return h_T

    # ---------------------------
    # PUBLIC: Forward mapping
    # ---------------------------
    def forward(self, x: torch.Tensor):
        """
        x → y
        """
        u = self.input_x(x)
        h = self._evolve(u)
        y_pred = self.forward_head(h)
        return y_pred

    # ---------------------------
    # PUBLIC: Inverse mapping
    # ---------------------------
    def inverse(self, y: torch.Tensor):
        """
        y → x
        """
        u = self.input_y(y)
        h = self._evolve(u)
        x_rec = self.inverse_head(h)
        return x_rec

    # ---------------------------
    # INTERNAL: for training use
    # ---------------------------
    def forward_with_latent(self, x: torch.Tensor):
        u = self.input_x(x)
        h = self._evolve(u)
        y_pred = self.forward_head(h)
        return y_pred, h

    def inverse_with_latent(self, y: torch.Tensor):
        u = self.input_y(y)
        h = self._evolve(u)
        x_rec = self.inverse_head(h)
        return x_rec, h
    
    from .trainer import train_bliq

    def fit(
        self,
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
        Train the BLiqNet model.
        """
        return train_bliq(
            self,
            x,
            y,
            epochs=epochs,
            lr=lr,
            lambda_rec=lambda_rec,
            lambda_inv=lambda_inv,
            lambda_latent=lambda_latent,
            verbose=verbose,
        )