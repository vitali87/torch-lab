import torch
from torch import nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

number_of_features = 5
n_samples = 1000
test_size = 0.2
data = datasets.make_classification(n_samples, number_of_features, n_informative=2)

X_train, X_test, y_train, y_test = train_test_split(
    data[0], data[1], test_size=test_size, random_state=42
)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(
                in_features=number_of_features, out_features=1, dtype=torch.float64
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_stack(x)


model = LogisticRegression()

ce_loss = nn.BCELoss()
optimiser = torch.optim.SGD(model.linear_stack.parameters(), lr=0.0002)


n_epochs = 10000
train_size = int(n_samples * (1 - test_size))
for i in range(n_epochs):
    y_pred = model(X_train)
    y_train_float = y_train.reshape(train_size, -1).double()
    step_loss = ce_loss(y_pred, y_train_float)
    optimiser.zero_grad()
    step_loss.backward()
    optimiser.step()
    print(f"epoch {i}, Loss: {step_loss.item()}")

for i in model.parameters():
    print(i)

y_pred = model(X_test)
test_size = int(n_samples * test_size)
y_test_float = y_test.reshape(test_size, -1).double()
print(ce_loss(y_pred, y_test_float))

y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]
print(f1_score(y_pred_binary, y_test))
