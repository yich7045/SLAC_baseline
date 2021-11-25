import torch.nn as nn
import torch

m = nn.Linear(20, 64)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
