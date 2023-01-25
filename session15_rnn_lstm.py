# Using RNN sentiment analysis

import numpy as np
import torch
from torch import nn
import pandas as pd
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv('AmazonReview.csv')
data.head()
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

# 1,2,3->negative(i.e 0)
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0

# 4,5->positive(i.e 1)
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

train_idx, test_idx, y_train, y_test = train_test_split(
    data.index, data["Sentiment"], test_size=0.2, stratify=data["Sentiment"])

y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

X_train = data.iloc[train_idx, 0]
X_test = data.iloc[test_idx, 0]

train_dataset = TensorDataset(torch.from_numpy(train_idx.to_numpy()), torch.from_numpy(y_train_np))
test_dataset = TensorDataset(torch.from_numpy(test_idx.to_numpy()), torch.from_numpy(y_test_np))

MAX_VOCAB_SIZE = 1_000
BATCH_SIZE = 16

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

D = []
[D.extend(doc.split()) for doc in X_train]

counts = Counter(D)
sorted_by_freq_tuples = sorted(counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples[:MAX_VOCAB_SIZE])

v1 = vocab(ordered_dict, specials=["<unk>", "<pad>"])
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
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, last_hidden):
        input_ = torch.cat((data, last_hidden), 1)
        hidden_ = self.i2h(input_)
        output_ = self.sigmoid(self.h2o(hidden_))
        return hidden_, output_

loss_fn = nn.MSELoss()
rnn = RNN(FULL_VOCAB_SIZE, FULL_VOCAB_SIZE, 1).to("cuda")
optimiser = torch.optim.SGD(rnn.parameters(), lr=0.001)

hidden = torch.zeros(BATCH_SIZE, FULL_VOCAB_SIZE).to("cuda")
target = torch.zeros(BATCH_SIZE, 1).to("cuda")

# max_len = 500

for train_indices, y_train in train_dataloader:

    X_train = data.iloc[train_indices, 0]

    V = [v1.lookup_indices(x.split()) for x in X_train]

    padded_sentences = pad_sequence([torch.tensor(p) for p in V], batch_first=True, padding_value=1)

    # for i in range(padded_sentences.shape[0]):
    #     padded_sentences[i] = nn.ConstantPad1d((0, max_len - padded_sentences[i].shape[0]), 1)(padded_sentences[i])

    TIMESTEPS = len(padded_sentences[0])

    G = torch.ones(BATCH_SIZE, TIMESTEPS, FULL_VOCAB_SIZE).to("cuda")
    for i in range(BATCH_SIZE):
        for j in range(TIMESTEPS):
            G[i, j] = torch.tensor(encoder.transform(padded_sentences[i, j].reshape(-1, 1))[0])

    for i in range(TIMESTEPS):
        hidden, output = rnn(G[:,i], hidden)
        if i == TIMESTEPS - 1:
            loss = loss_fn(output, target)
            print(f"TimeStep {i}, loss: {loss}")
            optimiser.zero_grad()
            loss.backward(retain_graph=True)
            optimiser.zero_grad()
            optimiser.step()
            print(optimiser.param_groups)