import torch
import torch.nn as nn

class HyperParameters:
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout=0.2, kernel_size=3, cnn_out_channels=16):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.cnn_out_channels = cnn_out_channels

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size}, {self.output_size}, {self.hidden_size}, {self.num_layers}, {self.dropout}, {self.kernel_size}, {self.cnn_out_channels})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.input_size == other.input_size and \
               self.output_size == other.output_size and \
               self.hidden_size == other.hidden_size and \
               self.num_layers == other.num_layers and \
               self.dropout == other.dropout and \
               self.kernel_size == other.kernel_size and \
               self.cnn_out_channels == other.cnn_out_channels
    
    def to_dict(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "kernel_size": self.kernel_size,
            "cnn_out_channels": self.cnn_out_channels
        }
    def from_dict(cls, d):
        return cls(**d)

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout=0.2, kernel_size=3, cnn_out_channels=[16,32,64]):
        super(CNN_LSTM_Model, self).__init__()
        # 一维卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels in cnn_out_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1))
            in_channels = out_channels
        # self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        # # 最大池化层（可选）
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()
        
    def forward(self, x, hidden=None):
        # x 的形状是 (batch_size, seq_len, input_dim)
        # 转换输入以符合 Conv1d 的要求，(batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        # 卷积操作
        for conv in self.conv_layers:
            x = conv(x)
            x = self.relu(x)
            x = self.pool(x)
        # 池化操作（如果需要）
        # x = self.pool(x)
        # 转换回 LSTM 的输入要求 (batch_size, seq_len, cnn_out_channels)
        x = x.transpose(1, 2)
        # LSTM 操作
        lstm_out, hidden = self.lstm(x, hidden)
        # 残差连接 （可选）
        # lstm_out = lstm_out + x
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        # 全连接层输出
        out = self.fc(out)
        return out, hidden
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)