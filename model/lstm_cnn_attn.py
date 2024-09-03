import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block
        
    def forward(self, x):
        return x + self.block(x)
    
class CNN_LSTM_ATTN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, kernel_size=3, out_channels=16, num_heads=4, dropout=0.1): 
        super(CNN_LSTM_ATTN, self).__init__()
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        # 最大池化层（可选）
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM 层
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        #多有头注意力
        self.attention= nn.MultiheadAttention(hidden_size, num_heads,dropout=dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        self.residual_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=1)
        )
        self.residual_lstm = nn.Linear(out_channels, hidden_size)
        self._init_weights()

    def forward(self, x):
        # 一维卷积层
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        # 卷积操作
        x = self.relu(x)
        # 池化操作（如果需要）
        x = self.pool(x)
        x = x + self.residual_conv(x)
        # 转换回 LSTM 的输入要求 (batch_size, seq_len, cnn_out_channels)
        x = x.transpose(1, 2)
        # LSTM 层
        lstm_out, _ = self.lstm(x)
        # 残差连接
        # lstm_out = lstm_out + x
        # 多有头注意力
        att_out, att_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # 残差连接
        # att_out = att_out + lstm_out
        out = att_out + self.residual_lstm(lstm_out)
        # 取最后一个时间步的输出
        out = att_out[:, -1, :]
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