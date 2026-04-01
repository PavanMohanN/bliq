import torch
import torch.nn as nn
from torchdiffeq import odeint
from .ode import LiquidODEFunc

class BLiqNet(nn.Module):
    """
    BLiqNet: Bidirectional Liquid Neural Network

    A continuous-time architecture designed for high-precision forward 
    mappings and stochastic inverse manifold reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        t_span: tuple = (0.0, 2.0),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Latent Projections
        self.input_x = nn.Linear(input_dim, hidden_dim)
        self.input_y = nn.Linear(output_dim, hidden_dim)
        
        # Stochastic Projector: Allows the model to explore solution 
        # manifolds in many-to-one (ill-posed) problems.
        self.noise_proj = nn.Linear(hidden_dim, hidden_dim)

        # Core Liquid Dynamics (LTC-based)
        self.ode_func = LiquidODEFunc(hidden_dim)

        # Output Heads
        self.forward_head = nn.Linear(hidden_dim, output_dim)
        self.inverse_head = nn.Linear(hidden_dim, input_dim)

        # Time Integration Horizon
        self.register_buffer(
            "t",
            torch.tensor([t_span[0], t_span[1]], dtype=torch.float32)
        )

    # ---------------------------
    # INTERNAL: Liquid Evolution
    # ---------------------------
    def _evolve(self, u_proj: torch.Tensor):
        """
        Evolves the latent state using the Liquid ODE function.
        RK4 is used to ensure stability and precision during the 
        continuous-time flow.
        """
        self.ode_func.set_input(u_proj)
        h0 = u_proj  # Data-dependent initial state
        
        # Integration via 4th-order Runge-Kutta
        h_t = odeint(self.ode_func, h0, self.t, method='rk4')

        return h_t[-1]

    # ---------------------------
    # PUBLIC: Forward (x → y)
    # ---------------------------
    def forward(self, x: torch.Tensor):
        """Standard mapping from input space to target space."""
        u = self.input_x(x)
        h = self._evolve(u)
        return self.forward_head(h)

    # ---------------------------
    # PUBLIC: Inverse (y → x)
    # ---------------------------
    def inverse(self, y: torch.Tensor, noise_level: float = 0.0):
        """
        Inverse mapping from target space back to input space.
        
        Args:
            y: Target tensor
            noise_level: Scale of latent perturbation to explore 
                         non-unique (many-to-one) solutions.
        """
        u = self.input_y(y)
        
        if noise_level > 0:
            # Inject stochasticity to break topological symmetry
            noise = torch.randn_like(u) * noise_level
            u = u + self.noise_proj(noise)
            
        h = self._evolve(u)
        return self.inverse_head(h)

    # ---------------------------
    # INTERNAL: Training Utilities
    # ---------------------------
    def forward_with_latent(self, x: torch.Tensor):
        """Used by the loss function to synchronize forward/inverse latents."""
        u = self.input_x(x)
        h = self._evolve(u)
        y_pred = self.forward_head(h)
        return y_pred, h

    def inverse_with_latent(self, y: torch.Tensor, noise_level: float = 0.05):
        """Used by the loss function to train the manifold sampler."""
        u = self.input_y(y)
        
        if noise_level > 0:
            noise = torch.randn_like(u) * noise_level
            u = u + self.noise_proj(noise)
            
        h = self._evolve(u)
        x_rec = self.inverse_head(h)
        return x_rec, h
    
    def fit(
        self,
        x,
        y,
        epochs: int = 1000,
        lr: float = 1e-3,
        lambda_rec: float = 0.01,
        lambda_inv: float = 2.0,
        lambda_latent: float = 0.2,
        verbose: bool = True,
    ):
        """
        High-level training API using Bidirectional Dual-Consistency loss.
        """
        from .trainer import train_bliq
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