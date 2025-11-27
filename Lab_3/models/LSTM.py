import torch

from torch import nn

from utils.vocab import Vocab

class LSTM(nn.Module):

    def __init__(self, vocab: Vocab):

        super(LSTM, self).__init__()

        self.embedding_dim = 256

        self.hidden_dim = 256

        self.num_layers = 5

        self.num_classes = vocab.total_labels

        self.vocab_size = vocab.vocab_size

        self.dropout = 0.4

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=vocab.pad_id)

        self.lstm = nn.LSTM(input_size=self.embedding_dim,

                            hidden_size=self.hidden_dim,

                            num_layers=self.num_layers,

                            batch_first=True,

                            dropout=self.dropout if self.num_layers > 1 else 0)

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):

        embedded = self.embedding(x)

        lstm_out, (hn, cn) = self.lstm(embedded)

        out = self.dropout_layer(hn[-1])

        out = self.fc(out)

        return out