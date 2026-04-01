# BLiqNet: Bidirectional Liquid Neural Networks

[](https://www.google.com/search?q=https://pypi.org/project/bliq/)
[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://github.com/PavanMohanN/bliq)

**BLiqNet** is a high-performance, continuous-time deep learning library designed to solve complex, ill-posed inverse problems using **Liquid Time-Constant (LTC) dynamics**. By synchronizing forward and inverse latent flows, BLiqNet can reconstruct entire solution manifolds where standard MLPs suffer from mode collapse.

-----

## 1\. Getting Started

### Installation

The library is managed via `pyproject.toml`. For the most stable experience and to access all internal utilities:

```bash
git clone https://github.com/PavanMohanN/bliq.git
cd bliq
pip install -e .
```

### Quickstart Example

Solve a non-linear mapping in just a few lines:

```python
import torch
from bliq.model import BLiqNet

# 1. Initialize for a 10D -> 2D problem
model = BLiqNet(input_dim=10, output_dim=2, hidden_dim=128)

# 2. Train using Bidirectional Dual-Consistency
model.fit(train_x, train_y, epochs=1000)

# 3. Forward Prediction
y_pred = model.forward(test_x)

# 4. Stochastic Inverse Sampling (Manifold Exploration)
# Generate multiple valid candidates for a single observation 'y'
with torch.no_grad():
    x_candidates = [model.inverse(y, noise_level=0.5) for _ in range(50)]
```

-----

## 2\. Core Features & "Why BLiq?"

  * **Liquid Latent Dynamics**: Unlike fixed-rate ODEs, BLiqNet uses **LTC formulation** where the network's "update speed" ($\tau$) adapts to the input complexity.
  * **Dual-Consistency Loss**: Enforces $y \approx f(g(y))$, ensuring that even in many-to-one mappings, every reconstructed $x$ remains physically consistent with the target $y$.
  * **Stochastic Symmetry Injection**: Uses latent perturbations to effectively "un-collapse" solutions, allowing the model to trace full geometric rings or spirals.
  * **General Purpose Engine**: Flexible architecture that scales from 2D toy problems to high-dimensional geodata or physics simulations.

-----

## 3\. Results: The Circle Benchmark

In the classic "Inverse Circle" ($y = x_1^2 + x_2^2$), standard models collapse to a single point. BLiqNet recovers the full manifold topology.

| Metric | MLP (Baseline) | **BLiqNet (Ours)** |
| :--- | :--- | :--- |
| **Forward $R^2$** | 0.9837 | **0.9984** |
| **Forward RMSE** | 0.2177 | **0.0677** |
| **Inverse Consistency** | High (Collapsed) | **Low Error (Structured)** |

-----

## 4\. Technical Requirements

  * **Core**: PyTorch 2.0+
  * **Solver**: `torchdiffeq` (supports RK4 and Dormand-Prince)
  * **Visualization**: Matplotlib & Scikit-learn (for metrics)

-----

## 5\. Citations & Research

If you use BLiqNet in your academic work, please cite both the official journal publication and the generalized framework:

### Official Publication

```bibtex
@article{neelamraju2026ground,
  title={Ground Motion Modelling with Bidirectional Liquid Neural Network (BLiqNet)},
  author={Neelamraju, Pavan Mohan and Raghukanth, STG},
  journal={Geodata and AI},
  pages={100080},
  year={2026},
  publisher={Elsevier}
}
```

### ArXiv Documentation (Coming Soon)

[Placeholder: Link to Generalized BLiqNet Documentation/Paper]

-----

## 6\. Community & License

  * **Contributing**: We welcome PRs\! See `CONTRIBUTING.md` for guidelines.
  * **License**: Licensed under the **MIT License**.
  * **Contact**: Open a GitHub Issue for bug reports or feature requests.

-----
