import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # Linear projections
        Q = self.Wq(x)  # (batch_size, seq_len, hidden_dim)
        K = self.Wk(x)  # (batch_size, seq_len, hidden_dim)
        V = self.Wv(x)  # (batch_size, seq_len, hidden_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        out = self.fc_out(attention_output)

        return out

class LSTM_Attention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout=0.2):
        super(LSTM_Attention, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
