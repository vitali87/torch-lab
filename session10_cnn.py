import torch
import torchvision
from torch import nn
import cv2 as cv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import transforms

n_classes = 10
AA = torchvision.datasets.MNIST("cnn/", train=True, download=True, transform=ToTensor())
BB = torchvision.datasets.MNIST("cnn/", train=False, download=True, transform=ToTensor())

# img0 = A.data[2].numpy()
# img1 = A.data[1].numpy()
# cv.imshow("Display window",img0 )
# cv.imshow("bla", img1)
#
# # window is displayed until we press any key
# k = cv.waitKey(0)
# cv.destroyAllWindows()

dty = torch.double


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(10, 10),
                dtype=dty
            ),
            nn.Flatten(),
            nn.LazyLinear(
                out_features=20, dtype=dty
            ),
            nn.LazyLinear(
                # in_features=20,
                out_features=n_classes, dtype=dty
            )
        )

    def forward(self, x):
        return self.stack(x)


model = SimpleCNN()
model.to("cuda")

ce_loss = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.stack.parameters(), lr=0.001)

batch_size = 64

train_dataloader = DataLoader(AA,
                              batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(BB,
                             batch_size=batch_size, shuffle=True)

val_losses = []
train_losses = []

n_epochs = 10
for i in range(n_epochs):
    for batch, train in enumerate(train_dataloader):
        x_train = train[0].double().to("cuda")
        y_train = train[1].long().to("cuda")
        y_pred = model(x_train)
        step_loss = ce_loss(y_pred, y_train)
        optimiser.zero_grad()
        step_loss.backward()
        optimiser.step()
        with torch.no_grad():
            for batch_test, test in enumerate(test_dataloader):
                x_test = test[0].double().to("cuda")
                y_test= test[1].long().to("cuda")
                if batch_test == 1:
                    break
            y_test_pred = model(x_test)
            test_loss = ce_loss(y_test_pred, y_test)
        print(f"epoch {i}, batch {batch}, Train Loss: {step_loss.item()}, Test Loss {test_loss.item()}")
        train_losses.append(step_loss.item())
        val_losses.append(test_loss.item())
#
for i in model.parameters():
    print(i)
#
# n_ = 20
# print(torch.round(y_test_pred[:n_]))
# print(y_test[:n_])
#
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()