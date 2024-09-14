import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x, hidden, cell):
        # LSTM 输出预测值
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_heads=8):
        super(AttentionDecoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch_size, 1, output_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)

        # 解码器的 LSTM 层
        lstm_output, (hidden, cell) = self.lstm(x, (hidden, cell))  # (batch_size, 1, hidden_size)

        # 注意力计算
        attn_output, attn_weights = self.attention(
            lstm_output.transpose(0, 1),  # query (seq_len, batch_size, hidden_size)
            encoder_outputs.transpose(0, 1),  # key (seq_len, batch_size, hidden_size)
            encoder_outputs.transpose(0, 1)  # value (seq_len, batch_size, hidden_size)
        )
        
        attn_output = attn_output.transpose(0, 1)  # (batch_size, 1, hidden_size)

        # 全连接层，输出最终结果
        prediction = self.fc(attn_output.squeeze(1))  # (batch_size, output_size)

        return prediction, hidden, cell
    
class Seq2SeqWithMultiheadAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithMultiheadAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_seq, target_seq_len):
        # 编码输入序列，获取编码器的隐藏状态
        encoder_outputs, hidden, cell = self.encoder(source_seq)

        # 初始化解码器的输入
        decoder_input = source_seq[:, -1:, :]  # 最后一个时间步作为初始输入

        outputs = []
        for t in range(target_seq_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(prediction.unsqueeze(1))  # 将输出结果保存
            decoder_input = prediction.unsqueeze(1)  # 递归使用预测结果作为下一时间步的输入

        outputs = torch.cat(outputs, dim=1)  # 将所有时间步的结果拼接
        return outputs
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_seq, target_seq_len):
        hidden, cell = self.encoder(source_seq)
        
        # 初始化 decoder input，通常是一个初始值
        decoder_input = source_seq[:, -1:, :]  # 取最后一个时间步作为 decoder 的初始输入
        outputs = []
        
        for t in range(target_seq_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            decoder_input = prediction  # 将预测值作为下一时间步的输入

        outputs = torch.cat(outputs, dim=1)  # 将所有时间步的预测结果拼接
        return outputs