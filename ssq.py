import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataModel

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    balls, diff = DataModel.load_ssq_blue_diff()
    diff_data = np.diff(balls.to_numpy())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(diff_data.reshape(-1, 1))
    time_step = 12  # Number of time steps to look back
    X, y = create_dataset(scaled_data, time_step)

    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    input_size = 1
    hidden_size = 64
    num_layers = 2

    model = LSTMModel(input_size, hidden_size, num_layers)

    num_epochs = 20
    learning_rate = 0.001
    batch_size = 32

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    last_window = torch.tensor(scaled_data[-time_step:].reshape(1, time_step, 1), dtype=torch.float32)

    # Predict next value
    with torch.no_grad():
        predicted_scaled_value = model(last_window).numpy()
        predicted_value = scaler.inverse_transform(predicted_scaled_value)
        
    print(f"Predicted next value: {predicted_value[0,0]}")

    print(f"Actual next value: {balls.iloc[-1] + predicted_value[0,0]}")