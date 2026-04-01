# BLiqNet: Bidirectional Liquid Neural Networks

[](https://pytorch.org/)
[](https://opensource.org/licenses/MIT)

**BLiqNet** is the official implementation of the **Bidirectional Liquid Neural Network**, a continuous-time architecture designed for high-precision forward mapping and stochastic inverse manifold reconstruction.

By integrating **Liquid Time-Constant (LTC) dynamics** with a bidirectional dual-consistency loss, BLiqNet overcomes the "Mode Collapse" common in standard MLPs when solving ill-posed, many-to-one inverse problems (e.g., $y = x_1^2 + x_2^2$).

-----

## Core Features

  * **Liquid Time-Constant (LTC) Dynamics**: dh/dt is governed by input-dependent conductance, allowing the network to adapt its update speed to data complexity.
  * **Bidirectional Dual-Consistency**: Synchronized forward ($x \to y$) and inverse ($y \to x$) pathways through a shared latent Liquid ODE.
  * **Stochastic Manifold Sampling**: Employs latent perturbations and symmetry injection to explore entire solution manifolds in ill-posed settings.
  * **Physics-Consistent**: Prioritizes inverse feasibility ($y \approx f(g(y))$), ensuring reconstructed solutions obey the underlying laws of the data.

-----

## Theory

BLiqNet evolves its latent state $h$ using the LTC formulation:

$$\frac{dh}{dt} = -\left[\frac{1}{\tau} + f(h, u)\right]h + f(h, u)A$$

This ensures the system is "Liquid"—its internal update speed effectively changes in response to the input $u$, providing superior stability and expressivity compared to standard Neural ODEs.

-----

## Installation

The library is managed via `pyproject.toml`. To install in "editable" mode for development:

```bash
git clone https://github.com/PavanMohanN/bliq.git
cd bliq
pip install -e .
```

-----

## Quick Start

### 1\. General Purpose Usage

```python
import torch
from bliq.model import BLiqNet

# Initialize for a 10D -> 2D problem
model = BLiqNet(input_dim=10, output_dim=2, hidden_dim=128)

# Train using bidirectional dual-consistency
model.fit(train_x, train_y, epochs=1000)
```

### 2\. Probabilistic Inverse Sampling

For problems with non-unique solutions, use the stochastic `inverse` method to sample the manifold:

```python
# Generate 50 distinct valid candidates for an observation 'y'
with torch.no_grad():
    samples = [model.inverse(y, noise_level=0.5) for _ in range(50)]
```

-----

## Results: The Circle Benchmark

In the classic "Inverse Circle" benchmark ($y = x_1^2 + x_2^2$), BLiqNet demonstrates superior performance by preserving the manifold topology.

| Metric | MLP (Baseline) | **BLiqNet (Ours)** |
| :--- | :--- | :--- |
| **Forward $R^2$** | 0.9837 | **0.9984** |
| **Forward RMSE** | 0.2177 | **0.0677** |
| **Inverse Consistency** | High Error (Collapsed) | **Low Error (Structured)** |

-----

## Project Structure

  * `bliq/model.py`: General-purpose BLiqNet architecture.
  * `bliq/model_circle.py`: Symmetry-aware refactor for circular manifolds.
  * `bliq/ode.py`: Core Liquid Time-Constant (LTC) dynamics.
  * `bliq/loss.py`: Bidirectional dual-consistency loss functions.
  * `bliq/trainer.py`: Gradient-clipped training loop.

-----

## Citation

If you use **BLiqNet** in your research, please cite the official implementation:

```bibtex
@software{Neelamraju_BLiqNet_2026,
  author = {Pavan Mohan Neelamraju},
  title = {BLiqNet: Bidirectional Liquid Neural Networks for Inverse Problems},
  url = {https://github.com/PavanMohanN/bliq},
  version = {1.0.0},
  year = {2026}
}
```

-----

## 许可证 (License)

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
