import torch

# simplest one-layer neural network
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(z.grad_fn)
print(loss.grad_fn)
loss.backward()
print(w.grad)
print(b.grad)

# Disabling Gradient Tracking
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# Reasons for disabling:
# 1) To mark some parameters in your neural network as frozen parameters. This is a very common scenario for fine-tuning
# a pretrained NN
# 2) To speed up computations when you are only doing forward pass, because computations on tensors that do not track
# gradients would be more efficient.
