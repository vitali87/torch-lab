import torch
from torch import nn

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)

print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))  # Or just X @ W_xh + H @ W_hh
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))


class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input_ = torch.cat((data, last_hidden), 1)
        hidden_ = self.i2h(input_)
        output_ = self.h2o(hidden_)
        return hidden_, output_


rnn = RNN(50, 20, 10)

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
for i in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    current_loss = loss_fn(output, target)
    loss += current_loss
    print(f"TimeStep {i}, Current loss: {current_loss}, Cumulative loss {loss.item()}")
loss.backward()