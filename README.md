# BLiqNet: Bidirectional Liquid Neural Networks

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![DOI: Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxxx-blue.svg)](https://zenodo.org/badge/latestdoi/xxxxxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-26XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/26XX.XXXXX)

**The first continuous-time liquid neural network capable of bidirectional flow.**

**BLiqNet** is a general-purpose inference engine designed to solve complex, many-to-one inverse problems without suffering from mode collapse. By integrating Liquid Time-Constant (LTC) dynamics with stochastic manifold sampling, BLiqNet learns the underlying topology of continuous data, allowing for high-precision forward mapping and probabilistic inverse reconstruction.

## 🚀 Quick Start (One-Line Install)

Complexity kills adoption. Install BLiqNet directly via pip:

```bash
pip install bliqnet
````

*(For development, clone the repository and run `pip install -e .`)*

### General Purpose Initialization & Training

```python
import torch
from bliq.model import BLiqNet

# 1. Initialize for a high-dimensional mapping problem (e.g., 10D -> 2D)
model = BLiqNet(input_dim=10, output_dim=2, hidden_dim=128)

# 2. Train using bidirectional dual-consistency flow
model.fit(train_x, train_y, epochs=1000, lambda_inv=2.0)

# 3. Probabilistic Inverse Sampling (Many-to-One Resolution)
with torch.no_grad():
    # Sample 50 distinct valid candidates for an observation 'y'
    samples = [model.inverse(y, noise_level=0.5) for _ in range(50)]
```

-----

## 🧠 Mathematical Core: The Liquid Mechanism

Unlike standard Neural ODEs, BLiqNet governs its latent state $h$ using the **Liquid Time-Constant (LTC)** formulation:

$$\frac{dh}{dt} = -\left[\frac{1}{\tau} + f(h, u)\right]h + f(h, u)A$$

This ensures the system is truly "Liquid"—its internal update speed effectively changes in response to the input $u$. By deploying this in a bidirectional flow paradigm, the model synchronizes forward ($x \to y$) and inverse ($y \to x$) pathways, allowing it to navigate singularities, self-intersections, and complex topologies (like 3D helices and Bernoulli Lemniscates) with extreme numerical stability.

-----

## 📊 Visual Proof: Manifold Learning

*Figure 1: BLiqNet successfully reconstructing a complex, non-linear manifold without mode collapse, demonstrating topological awareness.*

*Figure 2: The distribution of Liquid Time-Constants ($\tau$) within the network, proving the model dynamically adapts its internal "speed" to resolve multi-scale spatial features.*

-----

## 📝 Citation & Academic Use

To ensure the longevity of this framework, **BLiqNet** is officially indexed. If you use this software or architecture in your research, please cite the foundational publications and the software itself.

### 1\. The Official Journal Application

For the application of BLiqNet to geodata and ground motion:

```bibtex
@article{neelamraju2026ground,
  title={Ground Motion Modelling with Bidirectional Liquid Neural Network (BLiqNet)},
  author={Neelamraju, Pavan Mohan and Raghukanth, STG},
  journal={Geodata and AI},
  pages={100080},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.geoai.2026.100080}
}
```

### 2\. General Architecture & Documentation (arXiv)

For general architecture details, math, and the underlying bidirectional framework:

```bibtex
@article{neelamraju2026bliqnet,
  title={BLiqNet: A General Framework for Bidirectional Liquid Neural Networks},
  author={Neelamraju, Pavan Mohan},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026}
}
```

### 3\. Software & Code (Zenodo)

```bibtex
@software{bliqnet_software_2026,
  author       = {Pavan Mohan Neelamraju},
  title        = {PavanMohanN/bliq: v1.0.0 Release Candidate},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {[https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)}
}
```

-----

## 🤝 Contributing & License

We welcome contributions from the community to expand the capabilities of continuous-time deep learning.
This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
