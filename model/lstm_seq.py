import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input):
        output, (hidden, cell) = self.lstm(input)
        return output, hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output =  self.fc(output)
        return output, hidden, cell

    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # encoder的输出
        encoder_outputs, hidden, cell = self.encoder(src)

        # decoder的输入初始化为<SOS> token
        decoder_input = trg[:, 0:1]

        # teacher forcing
        outputs = torch.zeros(trg.shape).to(self.device)
        for t in range(1, trg.shape[1]):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = trg[:, t:t+1] if teacher_force else output.max(1)[1].unsqueeze(1)

        return outputs