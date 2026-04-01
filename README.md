# BLiq: Bidirectional Liquid Neural Networks

BLiq is a minimal PyTorch library for solving inverse problems using Bidirectional Liquid Neural Networks (BLiqNet).

---

## Installation

```bash
pip install bliq

Quick Example

import torch
from bliq import BLiqNet

# dummy data
x = torch.randn(1000, 2)
y = (x[:, 0]**2 + x[:, 1]**2).unsqueeze(1)

model = BLiqNet(input_dim=2, output_dim=1)
model.fit(x, y, epochs=500)

# forward prediction
y_pred = model.forward(x)

# inverse prediction
x_rec = model.inverse(y)