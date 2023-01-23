# Using RNN and LSTm for sentiment analysis

import torch
from torch import nn
import pandas as pd
from torch.nn import functional as F
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab

data = pd.read_csv('AmazonReview.csv')
data.head()
data.dropna(inplace=True)

# 1,2,3->negative(i.e 0)
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0

# 4,5->positive(i.e 1)
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

################## if using english words#########
tokenizer = get_tokenizer("basic_english")

tokens = [tokenizer(doc) for doc in data.Review]
##################################################

D = []
[D.extend(doc.split()) for doc in data.Review]

counts = Counter(D)
ordered_dict = counts.most_common()
counts.values()
counts.keys()

idx_l = [i for i in range(len(counts))]
idx_t = torch.tensor(idx_l)

vv = vocab(counts)
vv.lookup_tokens([342])
vv.lookup_indices(['thought'])

# voc = build_vocab_from_iterator(counts, specials=["<unk>"])
# voc.lookup_tokens([2])
#
# words = [voc.lookup_tokens([i])[0] for i in range(len(voc))]
# voc.lookup_indices(['$15'])
# voc.lookup_token(0)

A = F.one_hot(idx_t)


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