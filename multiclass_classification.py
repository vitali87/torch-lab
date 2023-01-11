import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split

number_of_features = 5
n_classes = 3
n_samples = 1000
test_size = 0.2
data = datasets.make_classification(n_samples,
                                    number_of_features,
                                    n_classes=n_classes,
                                    n_informative=3)

X_train, X_test, y_train, y_test = train_test_split(
    data[0], data[1], test_size=test_size, random_state=42
)

X_train = torch.from_numpy(X_train).to("cuda")
y_train = torch.from_numpy(y_train).to("cuda")

X_test = torch.from_numpy(X_test).to("cuda")
y_test = torch.from_numpy(y_test).to("cuda")

y_train = torch.nn.functional.one_hot(y_train)
y_test = torch.nn.functional.one_hot(y_test)


class MultiClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(
                in_features=number_of_features,
                out_features=4, dtype=torch.float64
            ),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(
                in_features=4,
                out_features=4, dtype=torch.float64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4,
                out_features=n_classes, dtype=torch.float64
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.stack(x)


model = MultiClass()
model.to("cuda")

ce_loss = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.stack.parameters(), lr=0.001)

from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

import matplotlib.pyplot as plt

val_losses = []
train_losses = []

n_epochs = 1000
train_size = int(n_samples * (1 - test_size))
for i in range(n_epochs):
    for batch, (X, y) in enumerate(train_dataloader):
        y_pred = model(X_train)
        y_train_float = y_train.reshape(train_size, -1).double()
        step_loss = ce_loss(y_pred, y_train_float)
        optimiser.zero_grad()
        step_loss.backward()
        optimiser.step()
        with torch.no_grad():
            y_test_pred = model(X_test)
            y_test_float = y_test_pred.double()
            test_loss = ce_loss(y_test_pred, y_test_float)
        print(f"epoch {i}, batch {batch}, Train Loss: {step_loss.item()}, Test Loss {test_loss.item()}")
        train_losses.append(step_loss.item())
        val_losses.append(test_loss.item())

for i in model.parameters():
    print(i)

n_ = 20
print(torch.round(y_test_pred[:n_]))
print(y_test[:n_])

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()