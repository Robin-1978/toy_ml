import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import DataModel
from torch.optim.lr_scheduler import StepLR

# 加载数据
table = DataModel.load_ssq_blue_one()
data = table['Ball_7'].values

# 数据删除最后一个
data = data[:-1]

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# 计算差分序列并进行归一化
diff = np.diff(data, prepend=data[0])
diff_scaler = MinMaxScaler(feature_range=(0, 1))
diff_normalized = diff_scaler.fit_transform(diff.reshape(-1, 1))

# 计算其他特征
def compute_features(data, diff):
    parity = (data % 2).astype(int)
    size_flag = (data > 8).astype(int)
    lag1 = np.roll(data, shift=1)
    lag2 = np.roll(data, shift=2)
    lag_diff1 = data - lag1
    lag_diff2 = data - lag2
    return parity, size_flag, lag1, lag2, lag_diff1, lag_diff2

parity, size_flag, lag1, lag2, lag_diff1, lag_diff2 = compute_features(data, diff)

# 特征归一化
size_flag_scaler = MinMaxScaler(feature_range=(0, 1))
lag1_scaler = MinMaxScaler(feature_range=(0, 1))
lag2_scaler = MinMaxScaler(feature_range=(0, 1))
lag_diff1_scaler = MinMaxScaler(feature_range=(0, 1))
lag_diff2_scaler = MinMaxScaler(feature_range=(0, 1))

size_flag_normalized = size_flag_scaler.fit_transform(size_flag.reshape(-1, 1))
lag1_normalized = lag1_scaler.fit_transform(lag1.reshape(-1, 1))
lag2_normalized = lag2_scaler.fit_transform(lag2.reshape(-1, 1))
lag_diff1_normalized = lag_diff1_scaler.fit_transform(lag_diff1.reshape(-1, 1))
lag_diff2_normalized = lag_diff2_scaler.fit_transform(lag_diff2.reshape(-1, 1))

# 创建滑动窗口数据
def create_sequences(data, diff, parity, size_flag, lag1, lag2, lag_diff1, lag_diff2, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        features = np.column_stack((
            data[i:i+seq_length],
            diff[i:i+seq_length],
            parity[i:i+seq_length],
            size_flag[i:i+seq_length],
            lag1[i:i+seq_length],
            lag2[i:i+seq_length],
            lag_diff1[i:i+seq_length],
            lag_diff2[i:i+seq_length]
        ))
        X.append(features)
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_sequences(data_normalized, diff_normalized, parity, size_flag_normalized, lag1_normalized, lag2_normalized, lag_diff1_normalized, lag_diff2_normalized, seq_length)

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# 创建DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, output_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型、定义损失函数和优化器
model = LSTMModel(input_size=8, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

best_loss = float('inf')
best_model_path = 'best_lstm_model.pth'
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}')

    # Save the model if the average loss is the best so far
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved with loss: {best_loss:.4f}')

# 预测第3001个数值
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
with torch.no_grad():
    last_window = np.column_stack((
        data_normalized[-seq_length:], 
        diff_normalized[-seq_length:], 
        parity[-seq_length:],
        size_flag_normalized[-seq_length:],
        lag1_normalized[-seq_length:], 
        lag2_normalized[-seq_length:], 
        lag_diff1_normalized[-seq_length:], 
        lag_diff2_normalized[-seq_length:]
    ))
    last_window_tensor = torch.FloatTensor(last_window).unsqueeze(0)
    predicted_value_normalized = model(last_window_tensor)
    
    # 将预测结果逆归一化
    predicted_value = scaler.inverse_transform(predicted_value_normalized.numpy())
    print(predicted_value)
