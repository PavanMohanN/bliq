import torch
from bliq.model import BLiqNet

# dummy data
x = torch.randn(100, 2)
y = torch.sum(x, dim=1, keepdim=True)

model = BLiqNet(2, 1)

model.fit(x, y, epochs=200)

y_pred = model.forward(x)
x_rec = model.inverse(y)

print(y_pred.shape)
print(x_rec.shape)

from bliq.utils import compute_metrics

r2, rmse = compute_metrics(y, y_pred)
print(r2, rmse)

