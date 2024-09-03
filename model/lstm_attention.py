import torch
import torch.nn as nn

class LSTM_Attention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(LSTM_Attention, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 残差连接 
        # lstm_out = lstm_out + x
        att_out, att_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # 残差连接
        # att_out = att_out + lstm_out
        # max_pool = torch.max(att_out, dim=1)[0]
        # avg_pool = torch.mean(att_out, dim=1)
        out = self.fc(att_out[:, -1, :])  # Use the output from the last time step
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
