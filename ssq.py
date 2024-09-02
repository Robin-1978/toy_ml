from logging import critical
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataModel
from torch.utils.data import DataLoader, TensorDataset


import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log(message):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} - {message}")

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

def create_dataset_single_6(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
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
    def __init__(self, input_size, hidden_size, num_layers, num_classes, droupout, embed_size, heads):
        super(LSTMWithSelfAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=droupout)
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
    def __init__(self, input_dim, lstm_hidden_dim, output_dim, num_layers=2, dropout=0.2, kernel_size=3, cnn_out_channels=16):
        super(CNN_LSTM_Model, self).__init__()

        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, stride=1, padding=1)
        
        # 最大池化层（可选）
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def forward(self, x):
        # x: (sequence_length, batch_size, input_dim)
        x = self.embedding(x)
        # x: (sequence_length, batch_size, hidden_dim)
        x = self.transformer(x, x)  # (sequence_length, batch_size, hidden_dim)
        x = self.fc(x[-1])  # Use the last token of the sequence for classification/regression
        return x
    
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
    dropout = 0.2
    embed_size = 64
    head_num = 4

    # model = LSTMModel(input_size, output_size, hidden_size, num_layers) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers) #13
    # model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, embed_size, head_num) #8
    model = CNN_LSTM_Model(input_size, hidden_size, output_size, num_layers, dropout, 3, 16)  #7
    
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

def SimpleSingle(num, last_id = 0, device = 'cpu'):
    balls, diff = DataModel.load_ssq_single_diff(num)
    # balls, diff = DataModel.load_fc3d_single_diff(num)

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
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # Shape: [samples, time steps, features]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 2
    dropout = 0
    embed_size = 64
    head_num = 4

    # model = LSTMModel(input_size, output_size, hidden_size, num_layers).to(device) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers).to(device) #13
    # model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, dropout, embed_size, head_num).to(device) #8 #6
    # model = CNN_LSTM_Model(input_size, 64, output_size, num_layers, 0, 3, 16).to(device)  #7 #5
    model = CNN_LSTM_Model(input_size, 128, output_size, num_layers, 0.2, 3, 16).to(device)  ### #3
    # model = CNN_LSTM_Model(input_size, 96, output_size, num_layers, 0.2, 3, 32).to(device)  ### #7
    
    num_epochs = 500
    learning_rate = 0.01
    batch_size = 32

    # Loss and optimizer
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
        
        if (epoch + 1) % 100 == 0:  # Print every 5 epochs
            log(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')

        if(scheduler.get_last_lr()[0] < 1e-5):
            log(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')
            break
    
    model.eval()
    last_sequence = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 1), dtype=torch.float32).to(device)
    predicted_diff_normalized = model(last_sequence).detach().cpu().numpy()
    print("Predicted difference normalized:", predicted_diff_normalized)
    
    # Reverse normalization
    predicted_diff = scaler.inverse_transform(predicted_diff_normalized)
    print("Predicted difference:", predicted_diff[0][0])
    
    # Reverse prediction
    last_observed_value = balls_data.iloc[last_id-1]
    predicted_value = last_observed_value + predicted_diff

    print(f'{num}: {last_observed_value} + {predicted_diff[0][0]} = {predicted_value[0][0]} -> {balls.iloc[last_id]}')
    return last_observed_value, predicted_diff[0][0], predicted_value[0][0], balls.iloc[last_id]

def SimpleSingle3d(num, last_id = 0):
    # balls, diff = DataModel.load_ssq_single_diff(num)
    balls, diff = DataModel.load_fc3d_single_diff(num)

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
    dropout = 0.2
    embed_size = 64
    head_num = 4

    # model = LSTMModel(input_size, output_size, hidden_size, num_layers) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers) #13
    model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, dropout, embed_size, head_num) #8
    # model = CNN_LSTM_Model(input_size, hidden_size, output_size, num_layers, dropout, 3, 16)  #7
    
    num_epochs = 1000
    learning_rate = 0.01
    batch_size = 64

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

    print(f'{num}: {last_observed_value} + {predicted_diff[0][0]} = {predicted_value[0][0]} -> {balls.iloc[last_id]}')
    
def Simple6(last_id = 0):
    balls, diff = DataModel.load_ssq_red_diff()

    # last_id = -10

    diff_data = diff.dropna().values
    balls_data = balls[1:]
    if(last_id == 0):
        last_id = len(diff_data)
    diff_data_train = diff_data[:last_id]
    # diff_data_train = diff_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(diff_data_train)
    
    time_step = 5  # Number of time steps to look back
    
    X, y = create_dataset_single_6(scaled_data, time_step)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    input_size = 6
    output_size = 6
    hidden_size = 128
    num_layers = 2
    dropout = 0
    embed_size = 64
    head_num = 4

    # model = LSTMModel(input_size, output_size, hidden_size, num_layers) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers) #13
    # model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, embed_size, head_num) #8
    model = CNN_LSTM_Model(input_size, hidden_size, output_size, num_layers, dropout, 3, 64)  #7
    
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
            log(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')

        if(scheduler.get_last_lr()[0] < 1e-5):
            break
    
    model.eval()
    last_sequence = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 6), dtype=torch.float32)
    predicted_diff_normalized = model(last_sequence).detach().numpy()
    print("Predicted difference normalized:", predicted_diff_normalized)
    
    # Reverse normalization
    predicted_diff = scaler.inverse_transform(predicted_diff_normalized)
    print("Predicted difference:", predicted_diff[0])
    
    # Reverse prediction
    last_observed_value = balls_data.iloc[last_id-1].values
    predicted_value = last_observed_value + predicted_diff[0]

    print(f'{last_observed_value} + {predicted_diff[0]} = {predicted_value} -> {balls.iloc[last_id].values}')

def create_dataset_class(data, classes, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(classes[i + time_step, 0])

    return np.array(X), np.array(y)
def SimpleClassifier(num, last_id = 0, device='cpu'):
    balls, diff = DataModel.load_ssq_single_diff(num)

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
    
    X, y = create_dataset_class(scaled_data, diff_data_train.reshape(-1,1)+15, time_step)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # Shape: [samples, time steps, features]
    y = torch.tensor(y, dtype=torch.long).to(device)

    input_size = 1
    output_size = 31
    hidden_size = 64
    num_layers = 2
    embed_size = 64
    head_num = 4

    # model = LSTMModel(input_size, output_size, hidden_size, num_layers).to(device) #
    # model = AttentionLSTMModel(input_size, output_size, hidden_size, num_layers).to(device) #13
    # model = LSTMWithSelfAttention(input_size, hidden_size, num_layers, output_size, embed_size, head_num).to(device) #8
    model = CNN_LSTM_Model(input_size, hidden_size, output_size).to(device)  #4
    # model = CNN_LSTM_Model(input_size, 96, output_size, num_layers, 0.2, 3, 32).to(device)  ### #7
    
    num_epochs = 500
    learning_rate = 0.01
    batch_size = 32

    # Loss and optimizer
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
        
        if (epoch + 1) % 100 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')

        if(scheduler.get_last_lr()[0] < 1e-5):
            break
    
    model.eval()
    last_sequence = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 1), dtype=torch.float32).to(device)
    predicted_diffs = model(last_sequence).detach().cpu()
    # predicted_diff = torch.max(predicted_diffs) -15
    predicted_diff = torch.argmax(predicted_diffs, dim=1) - 15
    # print("Predicted difference :", predicted_diff)

    # Reverse prediction
    last_observed_value = balls_data.iloc[last_id-1]
    predicted_value = last_observed_value + predicted_diff

    print(f'{last_observed_value} + {predicted_diff} = {predicted_value} -> {balls.iloc[last_id]}')
    return last_observed_value, predicted_diff.item(), predicted_value.item(), balls.iloc[last_id]

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

def PredictAtt(device = 'cpu'):
    session = DataModel.ConnectDB("data/att.db")

    for idx in range(-100, -1):
        base, diff, predict, goal = SimpleSingle(7, idx, device)
        row = DataModel.PredictTable()
        row.Basic = int(base)
        row.Step = diff
        row.Predict = predict
        row.Goal = int(goal)
        row.Diff = goal - predict
        if(goal - base < 0):
            if(diff < -0.5):
                row.Trend = 1
            else:
                row.Trend = 0
        elif (goal - base > 0):
            if(diff > 0.5):
                row.Trend = 1
            else:
                row.Trend = 0
        else:
            if(diff > -0.5 and diff < 0.5):
                row.Trend = 1
            else:
                row.Trend = 0

        # predict_row = DataModel.CreatePredictRow('att', base, diff, predict, goal, goal - predict)
        session.add(row)
        session.commit()
    session.close()

def PredictCnn(device):
    for idx in range(-1, -100, -1):
        # base, diff, predict, goal = SimpleSingle(7, idx, device)
        base, diff, predict, goal = SimpleClassifier(7, idx, device)
        session = DataModel.ConnectDB("data/cnn.db")
        row = DataModel.PredictTable()
        row.Basic = int(base)
        row.Step = diff
        row.Predict = predict
        row.Goal = int(goal)
        row.Diff = goal - predict
        if(goal - base < 0):
            if(diff < 0):
                row.Trend = 1
            else:
                row.Trend = 0
        elif (goal - base > 0):
            if(diff > 0):
                row.Trend = 1
            else:
                row.Trend = 0
        else:
            if(diff > -0.5 and diff < 0.5):
                row.Trend = 1
            else:
                row.Trend = 0
        session.add(row)
        for attemp in  range(5):
            try:
                session.commit()
            except:
                time.sleep(1)
        session.close()

if __name__ == '__main__':
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method('spawn')

    print(f"Using device: {device}")
    import argparse
    parser = argparse.ArgumentParser(description="SSQ arguments")
    # parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=500)
    parser.add_argument("-b", "--ball_num", type=int, help="Ball Number", default=7)
    parser.add_argument("-p", "--predict_num", type=int, help="Predict Number", default=0)
    args = parser.parse_args()
    # for i in range(-10,0):
    #     SimpleClassifier(i)    
    # Complex(-2)
    # for i in range(-10,0):
    #     Simple(i)
    # Simple6(-2)
    # Simple6(-2)
    # SimpleSingle(1, 0)
    # SimpleSingle(2, 0)
    # SimpleSingle(3, 0)
    # SimpleSingle(4, 0)
    # SimpleSingle(5, 0)
    # SimpleSingle(6, 0)
    # SimpleSingle(7, 0)
    # SimpleSingle(args.ball_num, args.predict_num)
    # SimpleSingle3d(args.ball_num, args.predict_num)
    # Simple(0)
    # PredictAtt()
    PredictCnn(device)
    # SimpleSingle(7, 0, device)
    # SimpleClassifier(0, device)