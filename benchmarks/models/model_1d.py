import math
import torch

from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, padding_idx):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.sum(x, dim=1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
            self, vocab_size, embedding_dim, max_length,
            num_heads, hidden_dim, output_dim, num_layers, dropout, padding_idx):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(max_length, embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)
        x = torch.sum(x, dim=1)
        x = self.fc(x)
        return x
