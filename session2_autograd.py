import torch

# autograd
x = torch.arange(4.0)
x.requires_grad_(True)

x.grad

y = 2 * torch.dot(x, x)

y.backward()
x.grad

# check that matches
x.grad == 4 * x

# Let's take another loss function
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()

# Backward for Non-Scalar Variables
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))# Faster: y.sum().backward()
x.grad

# Detaching Computation
# z = x * y and y = x * x
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u

# compute for y
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

# dynamic computational graph of pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
# it's a linear function
a.grad == d/a

