import torch
import torch.nn as nn

class LiquidODEFunc(nn.Module):
    """
    Implements True Liquid Time-Constant (LTC) Dynamics:
        dh/dt = -[1/tau + f(h, u)] * h + f(h, u) * A
    
    Where:
        - 1/tau is the system's passive decay.
        - f(h, u) is the input-dependent 'liquid' conductance.
    """

    def __init__(self, dim_h: int):
        super().__init__()

        self.dim_h = dim_h

        # Conductance networks (The 'Liquid' part)
        # f(h, u) determines how fast the state updates based on current input
        self.W = nn.Linear(dim_h, dim_h)  # Hidden weights
        self.B = nn.Linear(dim_h, dim_h)  # Input weights

        # Tau: The time-constant (clamped to be positive)
        self.tau = nn.Parameter(torch.ones(1, dim_h))
        
        # A: The bias/target state the system gravitates toward
        self.A = nn.Parameter(torch.zeros(1, dim_h))

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
        LTC ODE function evaluation.
        """
        if self.u is None:
            raise RuntimeError("Input u not set. Call set_input(u) before integration.")

        # 1. Calculate input-dependent conductance f(h, u)
        # This makes the time-constant 'liquid' because it changes with the input.
        f_hu = torch.sigmoid(self.W(h) + self.B(self.u))

        # 2. Apply the LTC formula:
        # dh/dt = -[1/tau + f_hu] * h + f_hu * A
        # We use softplus on tau to ensure it remains a positive time-constant.
        inv_tau = 1.0 / (nn.functional.softplus(self.tau) + 1e-6)
        
        dhdt = -(inv_tau + f_hu) * h + (f_hu * self.A)

        return dhdt