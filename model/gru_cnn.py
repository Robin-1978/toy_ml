import torch
import torch.nn as nn

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2, dropout=0.2, kernel_size=3, cnn_out_channels=16):
        super(CNN_GRU_Model, self).__init__()
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=1)
        # 最大池化层（可选）
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # gru 层
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._init_weights()
        
    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, input_dim)
        # 转换输入以符合 Conv1d 的要求，(batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        # 卷积操作
        x = torch.relu(self.conv1d(x))
        # 池化操作（如果需要）
        x = self.pool(x)
        # 转换回 LSTM 的输入要求 (batch_size, seq_len, cnn_out_channels)
        x = x.transpose(1, 2)
        # LSTM 操作
        gru_out, _ = self.gru(x)
        # 残差连接 （可选）
        # lstm_out = lstm_out + x
        # 取最后一个时间步的输出
        out = gru_out[:, -1, :]
        # 全连接层输出
        out = self.fc(out)
        return out
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)