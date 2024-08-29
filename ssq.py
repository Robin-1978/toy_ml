import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataModel
from torch.utils.data import DataLoader, TensorDataset

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

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    balls, diff = DataModel.load_ssq_blue_diff()

    last_id = -10

    diff_data = diff.dropna().values
    diff_data_train = diff_data[:last_id]
    # diff_data_train = diff_data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(diff_data_train.reshape(-1, 1))
    
    time_step = 5  # Number of time steps to look back
    
    X, y = create_dataset(scaled_data, time_step)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Shape: [samples, time steps, features]
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 3

    model = LSTMModel(input_size, output_size, hidden_size, num_layers)

    num_epochs = 100
    learning_rate = 0.001
    batch_size = 32

    # Loss and optimizer
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
        
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}')
    
    model.eval()
    last_sequence = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 1), dtype=torch.float32)
    predicted_diff_normalized = model(last_sequence).detach().numpy()
    print("Predicted difference normalized:", predicted_diff_normalized)
    
    # Reverse normalization
    predicted_diff = scaler.inverse_transform(predicted_diff_normalized)
    print("Predicted difference:", predicted_diff[0][0])
    
    # Reverse prediction
    last_observed_value = balls.iloc[last_id-1]
    predicted_value = last_observed_value + predicted_diff

    print(f'{last_observed_value} + {predicted_diff[0][0]} = {predicted_value[0][0]} -> {balls.iloc[last_id]}')
