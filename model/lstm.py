import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super(LSTM_Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def forward(self, x):
        out, _ = self.lstm(x)
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
