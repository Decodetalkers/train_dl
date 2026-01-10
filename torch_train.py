import torch

b = torch.tensor([1.0, 2.0, 3.0])

print(b)
print(b.dtype)
print(b.shape)

c = b.matmul(b)

print(c.dtype)
print(c.shape)
print(c)
