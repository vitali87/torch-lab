# Using RNN for sentiment analysis

import numpy as np
import torch
from torch import nn
import pandas as pd
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Hyper-params
VOCAB_SIZE = 1_000
TIMESTEPS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# DATA
data = pd.read_csv("AmazonReview.csv")
data.head()
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

# ENCODING
# 1,2,3->negative(i.e 0)
data.loc[data["Sentiment"] <= 3, "Sentiment"] = 0
# 4,5->positive(i.e 1)
data.loc[data["Sentiment"] > 3, "Sentiment"] = 1

# SPLIT
train_idx, test_idx, y_train, y_test = train_test_split(
    data.index, data["Sentiment"], test_size=0.2, stratify=data["Sentiment"]
)

y_train_np = np.array(y_train, dtype=np.double)
y_test_np = np.array(y_test, dtype=np.double)

X_train = data.iloc[train_idx, 0]
X_test = data.iloc[test_idx, 0]

train_dataset = TensorDataset(
    torch.from_numpy(train_idx.to_numpy()).double(), torch.from_numpy(y_train_np)
)
test_dataset = TensorDataset(
    torch.from_numpy(test_idx.to_numpy()).double(), torch.from_numpy(y_test_np)
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

D = []
[D.extend(doc.split()) for doc in X_train]

counts = Counter(D)
sorted_by_freq_tuples = sorted(counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples[:VOCAB_SIZE])

v1 = vocab(ordered_dict, specials=["<unk>", "<pad>", "<bos>", "<eos>"])
v1.set_default_index(v1["<unk>"])
idx = v1.get_stoi().values()
tokens = v1.get_itos()
vals = torch.from_numpy(np.fromiter(idx, dtype=int))

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(vals.reshape(-1, 1))

FULL_VOCAB_SIZE = len(v1)


class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.data_size = data_size

        self.input_size = self.data_size + self.hidden_size

        self.i2h = nn.Linear(self.input_size, self.hidden_size, dtype=torch.double)
        self.h2o = nn.Linear(self.hidden_size, self.output_size, dtype=torch.double)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_, last_hidden):
        input_ = torch.cat((data_, last_hidden), 1).double()
        hidden_ = self.i2h(self.sigmoid(input_))
        output_ = self.h2o(hidden_)
        return hidden_, output_

loss_fn = nn.BCEWithLogitsLoss()
rnn = RNN(FULL_VOCAB_SIZE, FULL_VOCAB_SIZE, 1).to("cuda")
optimiser = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

hidden = torch.ones(BATCH_SIZE, FULL_VOCAB_SIZE, dtype=torch.double).to("cuda")
G = torch.ones(BATCH_SIZE, TIMESTEPS, FULL_VOCAB_SIZE, dtype=torch.double).to("cuda")

PADS = ["<pad>"] * TIMESTEPS
counter = 0
losses = []
for train_indices, y_train in train_dataloader:
    if len(y_train) < BATCH_SIZE:
        break

    counter += 1
    y_train = y_train.to("cuda")

    X_train = data.iloc[train_indices, 0]

    V = [v1.lookup_indices(x[:TIMESTEPS].split()) for x in X_train]
    V.append(v1.lookup_indices(PADS))

    padded_sentences = pad_sequence(
        [torch.tensor(p) for p in V], batch_first=True, padding_value=1
    )
    for i in range(BATCH_SIZE):
        for j in range(TIMESTEPS):
            G[i, j] = torch.tensor(
                encoder.transform(padded_sentences[i][j].reshape(-1, 1))[0]
            )

    hidden2 = hidden
    for k in range(TIMESTEPS):
        hidden2, output = rnn(G[:, k], hidden2)
    loss = loss_fn(output, y_train.reshape(-1, 1))
    losses.append(loss.item())
    print(f"Train loop {counter}, loss: {loss:.3f}")
    optimiser.zero_grad()
    loss.backward(retain_graph=True)  # retain_graph=True
    optimiser.step()

plt.plot(losses)
print(optimiser.param_groups)
