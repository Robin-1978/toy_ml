import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataModel
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)


class LSTMStackModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTMStackModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_short = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.lstm_long = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self._init_weights()
        
    def forward(self, X_short, X_long):
        out_short, _ = self.lstm_short(X_short)
        out_long, _ = self.lstm_long(X_long)
        out = torch.cat((out_short, out_long), dim=-1)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
              

def create_dataset_single(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_size, 1) 
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_size)
        attn_weights = torch.tanh(self.attention(out))  # 计算注意力权重
        attn_weights = torch.softmax(attn_weights, dim=1)  # 归一化为概率
        context_vector = torch.sum(attn_weights * out, dim=1)  # 加权求和得到上下文向量

        out = self.fc(context_vector)

        return out

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Perform the dot-product attention calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # Scaled dot-product attention

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # (N, query_len, heads, head_dim) -> (N, query_len, embed_size)

        out = self.fc_out(out)
        return out
    
class LSTMWithSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embed_size, heads):
        super(LSTMWithSelfAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.self_attention = SelfAttention(embed_size, heads)
        self.fc = nn.Linear(hidden_size, num_classes)
        self._init_weights()
        
    def forward(self, x):
        h_lstm, _ = self.lstm(x)  # LSTM output
        attention_out = self.self_attention(h_lstm, h_lstm, h_lstm, mask=None)  # Apply Self-Attention
        out = self.fc(attention_out[:, -1, :])  # Use the output of the last time step
        return out
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)


class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, output_dim, kernel_size=3, cnn_out_channels=64):
        super(CNN_LSTM_Model, self).__init__()
        
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=1)
        
        # 最大池化层（可选）
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

        self._init_weights()
        
    def forward(self, x):
        # x 的形状是 (batch_size, seq_len, input_dim)
        
        # 转换输入以符合 Conv1d 的要求，(batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积操作
        x = F.relu(self.conv1d(x))
        
        # 池化操作（如果需要）
        x = self.pool(x)
        
        # 转换回 LSTM 的输入要求 (batch_size, seq_len, cnn_out_channels)
        x = x.transpose(1, 2)
        
        # LSTM 操作
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
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
def Simple(last_id = 0):
    balls, diff = DataModel.load_ssq_blue_diff()

    # last_id = -10

    diff_data = diff.dropna().values
    balls_data = balls[1:]
    if(last_id == 0):
        last_id = len(diff_data)
    diff_data_train = diff_data[:last_id]
    # diff_data_train = diff_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(diff_data_train.reshape(-1, 1))
    
    time_step = 5  # Number of time steps to look back
    
    X, y = create_dataset_single(scaled_data, time_step)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Shape: [samples, time steps, features]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 2
    embed_size = 64
    head_num = 4

    model = LSTMModel(input_size, output_size, hidden_size, num_layers) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers) #13
    # model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, embed_size, head_num) #8
    # model = CNN_LSTM_Model(input_size, hidden_size, output_size)  #7
    
    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 32

    # Loss and optimizer
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Create DataLoader for batch processing
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Step scheduler after each epoch
        loss = epoch_loss / len(data_loader)
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')

        if(scheduler.get_last_lr()[0] < 1e-5):
            break
    
    model.eval()
    last_sequence = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 1), dtype=torch.float32)
    predicted_diff_normalized = model(last_sequence).detach().numpy()
    print("Predicted difference normalized:", predicted_diff_normalized)
    
    # Reverse normalization
    predicted_diff = scaler.inverse_transform(predicted_diff_normalized)
    print("Predicted difference:", predicted_diff[0][0])
    
    # Reverse prediction
    last_observed_value = balls_data.iloc[last_id-1]
    predicted_value = last_observed_value + predicted_diff

    print(f'{last_observed_value} + {predicted_diff[0][0]} = {predicted_value[0][0]} -> {balls.iloc[last_id]}')

def create_dataset(diff_data, balls_data, time_step=1):
    X, y = [], []
    for i in range(len(diff_data) - time_step):
        features = np.column_stack((
            diff_data[i:i+time_step],
            balls_data[i:i+time_step],
        ))
        X.append(features)
        y.append(diff_data[i + time_step])
    return np.array(X), np.array(y)

def SimpleClassifier(last_id = 0):
    balls, diff = DataModel.load_ssq_blue_diff()


def Complex(last_id = 0):
    balls, diff = DataModel.load_ssq_blue_diff()

    # last_id = -10

    diff_data = diff.dropna().values
    balls_data = balls[1:].values
    if(last_id == 0):
        last_id = len(diff_data)
    diff_data_train = diff_data[:last_id]
    balls_data_train = balls_data[:last_id]
    # diff_data_train = diff_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(diff_data_train.reshape(-1, 1))

    scaled_balls = MinMaxScaler(feature_range=(-1, 1))
    scaled_balls_data = scaled_balls.fit_transform(balls_data_train.reshape(-1, 1))
    
    time_step = 5  # Number of time steps to look back
    
    X, y = create_dataset(scaled_data, scaled_balls_data, time_step)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)  # Shape: [samples, time steps, features]
    y = torch.tensor(y, dtype=torch.float32) #.unsqueeze(-1)

    input_size = 2
    output_size = 1
    hidden_size = 64
    num_layers = 2

    model = LSTMModel(input_size, output_size, hidden_size, num_layers)

    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 32

    # Loss and optimizer
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Create DataLoader for batch processing
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Step scheduler after each epoch
        loss = epoch_loss / len(data_loader)
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')
        
        if(scheduler.get_last_lr()[0] < 1e-5):
            break
    
    model.eval()
    features = np.column_stack((
        scaled_data[-time_step:],
        scaled_balls_data[-time_step:],
    )).reshape(1, time_step, 2)
    
    last_sequence = torch.tensor(features, dtype=torch.float32)

    predicted_diff_normalized = model(last_sequence).detach().numpy()
    print("Predicted difference normalized:", predicted_diff_normalized)
    
    # Reverse normalization
    predicted_diff = scaler.inverse_transform(predicted_diff_normalized)
    print("Predicted difference:", predicted_diff[0][0])
    
    # Reverse prediction
    last_observed_value = balls.iloc[last_id-1]
    predicted_value = last_observed_value + predicted_diff

    print(f'{last_observed_value} + {predicted_diff[0][0]} = {predicted_value[0][0]} -> {balls.iloc[last_id]}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SSQ arguments")
    parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=500)

    Simple(0)
    # Complex(-2)