import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

number_of_features = 5
n_samples = 1000
test_size = 0.2

data = datasets.make_regression(
    n_samples, number_of_features, n_informative=2, coef=True, bias=2
)

X_train, X_test, y_train, y_test = train_test_split(
    data[0], data[1], test_size=test_size, random_state=42
)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            in_features=number_of_features, out_features=1, dtype=torch.float64
        )

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()

mse_loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.002)

# Full batch SGD
n_epochs = 100
train_size = int(n_samples * (1 - test_size))
for i in range(n_epochs):
    y_pred = model(X_train)
    step_loss = mse_loss(y_pred, y_train.reshape(train_size, -1))
    optimiser.zero_grad()
    step_loss.backward()
    optimiser.step()
    y_pred = model(X_test)
    print(
        f"epoch {i}, Train Loss: {step_loss.item()}, Test R-Squared: {r2_score(y_test.detach().numpy(), y_pred.detach().numpy())}"
    )

# Estimated parameters
for i in model.parameters():
    print(i)

y_pred = model(X_test)
print(f"R^2: {r2_score(y_test.detach().numpy(), y_pred.detach().numpy())}")

# Real parameters were
print(data[2])
print("bias: 2")

# Mini-batch SGD batch
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for i in range(n_epochs):
    # training mini-batches
    for batch, (X, y) in enumerate(train_dataloader):
        y_pred = model(X)
        step_loss = mse_loss(y_pred, y.reshape(len(y), -1))
        optimiser.zero_grad()
        step_loss.backward()
        optimiser.step()
        print(f"epoch {i}, batch {batch}, Loss: {step_loss.item()}")
    # testing mini-batches
    with torch.no_grad():
        for batch, (X_t, y_t) in enumerate(test_dataloader):
            y_pred = model(X_t)
            print(
                f"epoch {i}, batch {batch}, r2_score: {r2_score(y_t.numpy(), y_pred.numpy())}"
            )

# mini-batch estimates
print("mini-batch estimates")
for i in model.parameters():
    print(i)
# Real parameters were
print("Real parameters")
print(data[2])
print("bias: 2")
