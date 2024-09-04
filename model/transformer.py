import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def forward(self, x):
        # x: (sequence_length, batch_size, input_dim)
        x = self.embedding(x)
        # x: (sequence_length, batch_size, hidden_dim)
        x = self.transformer(x)  # (sequence_length, batch_size, hidden_dim)
        # x = x.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim)
        x = x[:, -1, :]  # (batch_size, hidden_dim)
        x = self.fc(x)  # Use the last token of the sequence for classification/regression
        return x
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)