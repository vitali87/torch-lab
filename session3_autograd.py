import torch

# Some tensor methods
# https://pytorch.org/docs/stable/torch.html

# In-place operations for tensors
# https://pytorch.org/docs/stable/tensors.html#torch.Tensor.add_

x = torch.arange(3.0)
x.requires_grad_(True)
# Initialise x with some values
# x = torch.arange(3.0)


print(x.grad)

y = 2 * torch.dot(x, x)

# gradient of a scalar-valued function y with respect to a vector is vector-valued and has the same shape as x
y.backward()

# raises an error as we are calling backward twice on complicated function
# print(f" y print {y.backward()}")

print(x.grad)

# check that matches
print(x.grad == 4 * x)

# Let's take another loss function
# Reset the gradient as PyTorch does not automatically reset the gradient buffer when we record a new gradient
x.grad.zero_()
print(x.grad)

y = x.sum()

print(y.backward())
print(y.backward())
# Backward for Non-Scalar Variables
# deep learning frameworks vary in how they interpret gradients of non-scalar tensors
# pytorch raises error - need to reduce to scalar
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster version: y.sum().backward()
print(x.grad)

# Detaching Computation - removing calculation outside the recorded computational graph
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
