import torch

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

