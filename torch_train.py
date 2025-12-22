import torch

b = torch.tensor([1.,2.,3.])

print(b)
print(b.dtype)
print(b.shape)

c = b.matmul(b)

print(c.dtype)
print(c.shape)
print(c)
