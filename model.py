import torch
import torch.nn as nn
import torch.nn.init as init


class LSTMBlueModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5):
        super(LSTMBlueModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def process_inputs(self, x, y):
        hot_encodes = torch.nn.functional.one_hot(torch.tensor(x), num_classes=self.num_classes)
        hot_encodes = hot_encodes.reshape(hot_encodes.size(0), hot_encodes.size(1), -1).float()
        return hot_encodes, torch.tensor(y)

class LSTMRedModel(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_classes,
        num_layers=2,
        dropout=0.5,
    ):
        super(LSTMRedModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_classes, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size * input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes * input_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.size(0), x.size(1), -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def process_inputs(self, x, y):
        hot_encodes = torch.nn.functional.one_hot(torch.tensor(x), num_classes=self.num_classes)
        hot_encodes = hot_encodes.reshape(hot_encodes.size(0), hot_encodes.size(1), -1).float()
        return hot_encodes, torch.tensor(y)
    
class LSTMBallModel(nn.Module):
    def __init__(self, input_size, num_classes, output_size, hidden_size, num_layers=2, dropout=0.5):
        super(LSTMBallModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_size, num_classes * output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.out(lstm_out[:, -1, :])
        return out
    
    def process_inputs(self, x, y):
        return torch.tensor(x, dtype=torch.float), torch.tensor(y)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lstm
        # del self.fc1
        # del self.fc2
        # del self.fc3
        del self.out


class LSTMEmbedBallModel(nn.Module):
    def __init__(self, input_size, num_classes, embedding_size, hidden_size, num_layers=2, dropout=0.5):
        super(LSTMEmbedBallModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_classes, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size * input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, num_classes * input_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.reshape(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def process_inputs(self, x, y):
        return torch.tensor(x), torch.tensor(y)    
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.embedding
        del self.lstm
        del self.fc

class TransformerBallModel(nn.Module):
    def __init__(self, input_size, num_classes, nhead=4,num_layers=2, dropout=0.0):
        super(TransformerBallModel, self).__init__()

        self.d_model = input_size * num_classes
        self.num_layers = num_layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead, dropout=dropout
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(self.d_model, input_size * num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out