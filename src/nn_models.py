import torch
import torch.nn as nn


class LanguageModelLSTM(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(p=0.1)
        self.non_lin = nn.Tanh()


    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_batch)
        x, _ = self.rnn(embeddings)
        x = self.non_lin(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.non_lin(x)
        output = self.out(x)
        return output
    

class LanguageModelGRU(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(p=0.1)
        self.non_lin = nn.Tanh()

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_batch)
        x, _ = self.rnn(embeddings)
        x = self.non_lin(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.non_lin(x)
        output = self.out(x)
        return output
