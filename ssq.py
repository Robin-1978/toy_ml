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
from model.lstm_attention import LSTM_Attention
from model.lstm import LSTM_Model
from model.gru import GRU_Model
from model.lstm_cnn import CNN_LSTM_Model
from model.gru_cnn import CNN_GRU_Model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

def log(message):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} - {message}")

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def forward(self, x):
        # x: (sequence_length, batch_size, input_dim)
        x = self.embedding(x)
        # x: (sequence_length, batch_size, hidden_dim)
        x = self.transformer(x)  # (sequence_length, batch_size, hidden_dim)
        # x = x.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim)
        x = x[:, -1, :]  # (batch_size, hidden_dim)
        x = self.fc(x)  # Use the last token of the sequence for classification/regression
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

    # model = LSTM_Model(input_size, output_size, hidden_size num_layers, dropout=0).to(device) #
    # model = GRU_Model(input_size, hidden_size, output_size, num_layers, dropout=0).to(device)
    # model = CNN_LSTM_Model(input_size, output_size, 64, 2, 0, 3, 16).to(device)  #7 #5,3
    # model = CNN_LSTM_Model(input_size, output_size, 96, num_layers, 0.2, 3, 32).to(device)  ### #4,5
    # model = LSTM_Attention(1, 1, 64, 2, 4, 0).to(device)
    model = CNN_GRU_Model(input_size, output_size, 64, 2, 0, 3, 16).to(device)  #7 #5,3
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
    # model = CNN_LSTM_Model(input_size, hidden_size, output_size, num_layers, dropout, 3, 64)  #7
    model = LSTM_Attention(1, 1, 64, 2, 2, 0)
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
    # model = CNN_LSTM_Model(input_size, hidden_size, output_size).to(device)  #4
    model = CNN_LSTM_Model(input_size, 96, output_size, num_layers, 0.2, 3, 32).to(device)  ### #4
    # model = TransformerModel(input_size, 64, output_size, 4, 2, 0.2).to(device) #4
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
        
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')

        if(scheduler.get_last_lr()[0] < 1e-5):
            break
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')
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

    model = LSTM_Model(input_size, output_size, hidden_size, num_layers)

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
        base, diff, predict, goal = SimpleSingle(7, idx, device)
        # base, diff, predict, goal = SimpleClassifier(7, idx, device)
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

def predict_all():
    for ball_num in range(1, 8):
        set_seed(42)
        SimpleSingle(ball_num, 0, device)

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
    # SimpleSingle(args.ball_num, args.predict_num)
    # SimpleSingle3d(args.ball_num, args.predict_num)
    # Simple(0)
    # PredictAtt()
    # PredictCnn(device)
    # predict_all()
    SimpleSingle(args.ball_num, args.predict_num)