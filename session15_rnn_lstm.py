# Using RNN and LSTm for sentiment analysis
import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.nn import functional as F
from collections import Counter, OrderedDict
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('AmazonReview.csv')
data.head()
data.dropna(inplace=True)

# 1,2,3->negative(i.e 0)
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0

# 4,5->positive(i.e 1)
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

################## if using english words#########

# tokenizer = get_tokenizer("basic_english")
#
# tokens = [tokenizer(doc) for doc in data.Review]
# voc = build_vocab_from_iterator(tokens, specials=["<unk>"])
# voc.lookup_tokens([0])
# voc.lookup_indices(["it"])

##################################################
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

D = []
[D.extend(doc.split()) for doc in data['Review']]

counts = Counter(D)
sorted_by_freq_tuples = sorted(counts.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples[:MAX_VOCAB_SIZE])
v1 = vocab(ordered_dict, specials=["<unk>", "<pad>"])
v1.set_default_index(v1["<unk>"])
idx = v1.get_stoi().values()
vals = torch.from_numpy(np.fromiter(idx, dtype=int))

encoder = OneHotEncoder()
encoder.fit(vals.reshape(-1, 1))

V = [v1.lookup_indices(x.split()) for x in data.Review]
padded_sentences = pad_sequence([torch.tensor(p) for p in V], batch_first=True, padding_value=1)

encoder.transform(torch.tensor(padded_sentences).reshape(-1, 1))

A = F.one_hot(padded_sentences)

# stoi = v1.get_stoi()
# [stoi[word] for word in data.Review[0].split()]
# idx_l = [i for i in range(len(counts))]
# idx_t = torch.tensor(idx_l).to('cuda')

# vv = vocab(counts)sorted_by_freq_tuples = sorted(counts.items(), key=lambda x: x[1], reverse=True)
# ordered_dict = OrderedDict(sorted_by_freq_tuples[:25_000])
# v1 = vocab(ordered_dict, specials=["<unk>", "<pad>"])
# v1.set_default_index(v1["<unk>"])

# vv.lookup_tokens([342])
# vv.lookup_indices(['thought'])

# vocab(counts)
# voc = build_vocab_from_iterator(data.Review, specials=["<unk>"], max_tokens=MAX_VOCAB_SIZE)
# voc.lookup_tokens([2])
#
# words = [voc.lookup_tokens([i])[0] for i in range(len(voc))]
# voc.lookup_indices(['$15'])
# voc.lookup_token(0)

# A = F.one_hot(idx_t, num_classes=len(idx_t)-1)


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