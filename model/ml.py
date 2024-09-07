import torch
import torch.nn as nn

class MLModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64], dropout=0.1):
        super(MLModel, self).__init__()

        self.layers = nn.ModuleList()
        in_features = input_size
        # Create the hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        # self.pool = nn.AdaptiveAvgPool1d(1)  # Pool along the time dimension
        self.activation = nn.ReLU()  # Add activation function
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        self._init_weights()

    def forward(self, x, hidden=None):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)  # Apply activation function
            x = self.dropout(x)  # Apply dropout

        x = self.output_layer(x)
        return x[:, -1, :], hidden

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)