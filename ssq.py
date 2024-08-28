import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataModel

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        # 将所有列作为特征，包括新创建的特征
        X.append(data.iloc[i:(i + time_step), :].values)
        # 目标变量为下一个时间步的 'Ball_7'
        y.append(data.iloc[i + time_step, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    features = DataModel.load_ssq_blue_one()
    # features.drop(index=0) # frop the first row for NaN
    features = features[1:-1]
    scaler_ball = MinMaxScaler(feature_range=(0, 1))
    scaler_diff = MinMaxScaler(feature_range=(0, 1))
    features['Ball_7'] = scaler_ball.fit_transform(features['Ball_7'].values.reshape(-1, 1))
    features['diff'] = scaler_diff.fit_transform(features['diff'].values.reshape(-1, 1))

    time_step = 12  # Number of time steps to look 
    
    X, y = create_dataset(features, time_step)

    input_size = 4
    output_size = 1
    hidden_size = 64
    num_layers = 2
    
    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], input_size)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    model = LSTMModel(input_size, output_size, hidden_size, num_layers)

    num_epochs = 100
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
    last_window = torch.tensor(features[-time_step:].to_numpy().reshape(1, time_step, input_size), dtype=torch.float32)

    # Predict next value
    with torch.no_grad():
        predicted_scaled_value = model(last_window).numpy()
        predicted_value = scaler_ball.inverse_transform(predicted_scaled_value)
        
    print(f"Predicted next value: {predicted_value[0,0].round().astype(int)}")

    # print(f"Actual next value: {features['Ball_7'].iloc[-1] + predicted_value[0,0]}")