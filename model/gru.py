import torch
import torch.nn as nn


class GRU_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        out = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(out)  # (batch_size, output_dim)
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
