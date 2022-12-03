import torch
import numpy as np

data = [[1,4], [4,7]]
x_data = torch.tensor(data)

np_array = np.array(data)

x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_np)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_np, dtype=torch.float)

shape = (2,3,)


rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(zeros_tensor.dtype)
print(zeros_tensor.device)
print(zeros_tensor.shape)

torch.numel(zeros_tensor)

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = torch.ones(4, 4).to("cuda")

tensor[0]
tensor[:, 0]
tensor[..., -1]

tensor[:,1] = 0

torch.cat([tensor, tensor, tensor], dim=1)

tensor1 = 4 *  tensor
y1 = tensor @ tensor1

z1 = tensor * tensor1

agg = tensor.sum()
agg_item = agg.item()

# autograd
x = torch.arange(4.0)
x.requires_grad_(True)

x.grad

y = 2 * torch.dot(x, x)

y.backward()
x.grad