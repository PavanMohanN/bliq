import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from .ode import LiquidODEFunc

class BLiqNet(nn.Module):
    """
    Bidirectional Liquid Neural Network (Symmetry-Aware Refactor)

    Core updates for Circle Manifold:
        - Angular Noise Injection: Converts Gaussian noise into polar coordinates.
        - Symmetry Breaking: Allows the ODE to explore the full 360-degree rotation.
        - Extended Time Horizon: Increased t_span for complex latent navigation.
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

        # Input projections
        self.input_x = nn.Linear(input_dim, hidden_dim)
        self.input_y = nn.Linear(output_dim, hidden_dim)
        
        # Noise projection for inverse diversity (Maps 2D polar coords to hidden_dim)
        self.noise_proj = nn.Linear(2, hidden_dim)

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
        Evolves latent state using RK4. 
        Higher t_span (2.0) allows the 'liquid' state to fully 'relax' 
        into the circular geometry.
        """
        self.ode_func.set_input(u_proj)
        h0 = u_proj  
        
        # Using rk4 for high-fidelity manifold tracking
        h_t = odeint(self.ode_func, h0, self.t, method='rk4')

        return h_t[-1]

    # ---------------------------
    # PUBLIC: Forward mapping (x → y)
    # ---------------------------
    def forward(self, x: torch.Tensor):
        u = self.input_x(x)
        h = self._evolve(u)
        return self.forward_head(h)

    # ---------------------------
    # PUBLIC: Inverse mapping (y → x)
    # ---------------------------
    def inverse(self, y: torch.Tensor, noise_level: float = 0.0):
        """
        Inverts y to x using Angular Injection.
        By randomizing the starting 'angle' in the latent space, we break
        the topological bias of the ODE and reconstruct the full circle.
        """
        u = self.input_y(y)
        
        if noise_level > 0:
            # 1. Generate random angles [0, 2pi]
            batch_size = y.size(0)
            angle = torch.rand(batch_size, 1, device=y.device) * 2 * np.pi
            
            # 2. Create polar coordinates (The Symmetry Compass)
            # This tells the model WHICH side of the circle to flow toward.
            u_noise = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
            
            # 3. Inject into latent space
            u = u + self.noise_proj(u_noise) * noise_level
            
        h = self._evolve(u)
        return self.inverse_head(h)

    # ---------------------------
    # INTERNAL: Training Utilities
    # ---------------------------
    def forward_with_latent(self, x: torch.Tensor):
        u = self.input_x(x)
        h = self._evolve(u)
        y_pred = self.forward_head(h)
        return y_pred, h

    def inverse_with_latent(self, y: torch.Tensor, noise_level: float = 0.2):
        """
        During training, we use a higher noise level (0.2) to force
        the Liquid ODE to see the many-to-one mapping immediately.
        """
        u = self.input_y(y)
        
        if noise_level > 0:
            batch_size = y.size(0)
            angle = torch.rand(batch_size, 1, device=y.device) * 2 * np.pi
            u_noise = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
            u = u + self.noise_proj(u_noise) * noise_level
            
        h = self._evolve(u)
        x_rec = self.inverse_head(h)
        return x_rec, h
    
    def fit(
        self,
        x,
        y,
        epochs: int = 1500, # Increased epochs for complex manifold convergence
        lr: float = 1e-3,
        lambda_rec: float = 0.01, # Lowered to avoid 'point-magnet' effect
        lambda_inv: float = 2.0,  # Highly weighted for cycle consistency
        lambda_latent: float = 0.2,
        verbose: bool = True,
    ):
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