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
    
class AttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_heads=8):
        super(AttentionDecoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch_size, 1, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)

        # LSTM 层
        lstm_output, (hidden, cell) = self.lstm(x, (hidden, cell))  # (batch_size, 1, hidden_size)

        # 注意力机制
        attn_output, attn_weights = self.attention(
            lstm_output.transpose(0, 1),  # query (seq_len, batch_size, hidden_size)
            encoder_outputs.transpose(0, 1),  # key (seq_len, batch_size, hidden_size)
            encoder_outputs.transpose(0, 1)  # value (seq_len, batch_size, hidden_size)
        )
        
        attn_output = attn_output.transpose(0, 1)  # (batch_size, 1, hidden_size)

        # 全连接层，预测
        prediction = self.fc(attn_output)  # (batch_size, 1, output_size)

        return prediction, hidden, cell

    
class Seq2SeqWithMultiheadAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithMultiheadAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_seq, target_seq_len):
        # 编码器
        encoder_outputs, hidden, cell = self.encoder(source_seq)

        # 解码器的初始输入
        decoder_input = encoder_outputs[:, -1:, :]  # (batch_size, 1, hidden_size)

        outputs = []
        for t in range(target_seq_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(prediction)  # 保存预测结果
            decoder_input = prediction  # 使用预测结果作为下一时间步的输入

        outputs = torch.cat(outputs, dim=1)  # 合并所有时间步的结果
        return outputs
