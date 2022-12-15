import torch
import numpy as np

data = [[1, 4], [4, 7]]
x_data = torch.tensor(data)

np_array = np.array(data)

x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_np)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_np, dtype=torch.float)

shape = (
    2,
    3,
)


rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(zeros_tensor.dtype)
print(zeros_tensor.device)
print(zeros_tensor.shape)

torch.numel(zeros_tensor)

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = torch.rand(4, 4).to("cuda")

print(tensor[0])
print(tensor[:, 0])
print(tensor[..., -1])


tensor[:, 1] = 0

t_cat_1 = torch.cat([tensor, tensor, tensor], dim=1)
t_cat_0 = torch.cat([tensor, tensor, tensor], dim=0)

tensor1 = 4 * tensor

# matrix multiplication
y1 = tensor @ tensor1

# element-wise multiplication
z1 = tensor * tensor1

agg = tensor.sum()
agg_item = agg.item()

# autograd
x = torch.arange(4.0)
x.requires_grad_(True)

print(x.grad)

y = 2 * torch.dot(x, x)

y.backward()
print(x.grad)

# check that matches
print(x.grad == 4 * x)

# Let's take another loss function
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()

# Backward for Non-Scalar Variables
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster version: y.sum().backward()
print(x.grad)

# Detaching Computation
# z = x * y and y = x * x
x.grad.zero_()
y = x * x

# separate a tensor from the computational graph. Returns a new tensor that doesn't require a gradient
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

# compute for y
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


# dynamic computational graph of pytorch
def f(a):
    b = a * 2
    # matrix or vector norm condition
    while b.norm() < 1000:
        b = b * 2
    return b if b.sum() > 0 else 100 * b


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# it's a linear function
print(a.grad == d / a)

