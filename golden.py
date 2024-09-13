import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

import pandas as pd
data = pd.read_csv('gold_price_data.csv')


# 仅使用 "Close" 列 (收盘价)
data = data[['Close']].dropna()
# 使用 MinMaxScaler 将数据缩放到 (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
scaled_data = np.expand_dims(scaled_data, axis=-1)
# 生成训练集和测试集
train_size = int(len(scaled_data) * 0.8)  # 使用80%的数据作为训练集
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 创建函数，将数据转换为LSTM输入格式
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 5  # 每个输入窗口的大小为60天
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 将数据转换为 PyTorch 张量
X_train = torch.from_numpy(X_train).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

# Step 2: 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 定义模型、损失函数和优化器
model = LSTMModel()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # 将训练数据变成 [batch_size, time_step, features] 格式
    outputs = model(X_train)
    optimizer.zero_grad()
    
    # 计算损失
    loss = criterion(outputs, y_train)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: 预测
model.eval()
train_predict = model(X_test)

# 反向缩放预测值到原始范围
train_predict = train_predict.detach().numpy()
train_predict = scaler.inverse_transform(train_predict)
y_test = y_test.detach().numpy()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 打印测试集的预测结果
print(f"Predicted prices: {train_predict[:5].flatten()}")
print(f"Actual prices: {y_test[:5].flatten()}")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(train_predict, label='Predicted Prices', color='red')
plt.title('Gold Price Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
# Step 1: 获取黄金价格数据
# gold_data = yf.Ticker("GLD")
# hist = gold_data.history(period="5y")  # 获取5年内的价格数据
# hist.to_csv('gold_price_data.csv')

# # 仅使用 "Close" 列 (收盘价)
# data = hist[['Close']].dropna()


# 展示
# plt.plot(data)
# plt.show()

