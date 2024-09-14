import yfinance as yf
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

# ticket = yf.Ticker("XAUUSD=X")
# gold_data = yf.download("XAUUSD=X", period="max")
# gold_data.save('gold_price_data_30.csv')

import DataModel
from utils import *
from model import lstm_attention, lstm_seq


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_start_method("spawn")
torch.autograd.set_detect_anomaly(True)

print(f"Using device: {device}")
# 仅使用 "Close" 列 (收盘价)
# data = data[['Close']].dropna()
# # 使用 MinMaxScaler 将数据缩放到 (0, 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)
# scaled_data = np.expand_dims(scaled_data, axis=-1)
# 生成训练集和测试集

time_step = 3

df, scaler = DataModel.load_gold_features(6)


features=[
    'Open_scale',
    'High_scale',
    'Low_scale',
    'Close_scale',
    'Volume_scale',
    'Close_mean_scale',
    'Close_std_scale',
    'Close_rsi_scale',
    'Close_zscore_scale',
]
targets=[
    "Close_scale",
]

split = 0.95
X, y, PX = PrepareData(df, features=features, targets=targets, window_size=3)
split_index =int(len(X) * split)
train_x = X[:split_index]
train_y = y[:split_index]
test_x = X[split_index:]
test_y = y[split_index:]

# Convert to PyTorch tensors
X_train = torch.tensor(train_x, dtype=torch.float32).to(device)
y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
X_test = torch.tensor(test_x, dtype=torch.float32).to(device)
y_test = torch.tensor(test_y, dtype=torch.float32).to(device)
X_predict = torch.tensor(PX, dtype=torch.float32).to(device)


batch_size = 16
num_epochs = 50
model = lstm_attention.LSTM_Attention(len(features), len(targets), hidden_size=64, num_layers=2, num_heads=16, dropout=0.1).to(device)
# model = lstm_seq.Seq2SeqWithMultiheadAttention(encoder=lstm_seq.Encoder(input_size=len(features), hidden_size=64), decoder=lstm_seq.AttentionDecoder(output_size=len(targets), hidden_size=64, num_heads=16)).to(device)
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
# criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

dataset = TensorDataset(X_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=num_epochs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

for epoch in range(num_epochs):
    model.train()
    hidden = None
    epoch_loss = 0
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()   
        if hidden is not None:
            h, c = hidden
            if h.size(1) != batch_X.size(0):  
                h = h[:, :batch_X.size(0), :].contiguous()
                c = c[:, :batch_X.size(0), :].contiguous()
                hidden = (h, c)
        outputs, hidden = model(batch_X, hidden)
        # outputs = model(batch_X, len(batch_y))
        hidden = (hidden[0].detach(), hidden[1].detach())
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Step scheduler after each epoch
    loss = epoch_loss / len(data_loader)
    scheduler.step(loss)
    # scheduler.step()

    if (epoch + 1) % 1 == 0:  # Print every 5 epochs
        log(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.8f} learning rate: {scheduler.get_last_lr()[0]}"
        )

    if scheduler.get_last_lr()[0] < 1e-5:
        log(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.8f} learning rate: {scheduler.get_last_lr()[0]}"
        )
        break

    model.eval()
    outputs, _= model(X_test)

    loss = criterion(outputs, y_test)

    if (epoch + 1) % 1 == 0:  # Print every 5 epochs
        log(
            f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {loss:.8f}"
        )

# def predict_future(model, last_data, future_steps=30):
#     model.eval()
#     predictions = []
#     current_data = last_data

#     for _ in range(future_steps):

#         # current_data_tensor = torch.from_numpy(current_data).type(torch.Tensor).unsqueeze(0)

#         next_pred, _ = model(current_data)

#         next_pred_value = next_pred.detach().numpy().reshape(-1)
#         predictions.append(next_pred_value[0])

#         current_data.append(next_pred_value)
#         current_data = current_data[1:]

#     return predictions


# # last_data = train_data[-time_step:]


# future_steps = len(y_test)
# future_predictions = predict_future(model, X_test[:1], future_steps)

# print(f"Future predicted prices: {future_predictions.flatten()}")

with torch.no_grad():
    model.eval()
    outputs, _= model(X_test)

    future_predictions = scaler['Close'].inverse_transform(outputs[-len(y_test):].cpu().detach().numpy())
    y_test = scaler['Close'].inverse_transform(y_test.cpu().detach().numpy())

    last, _ = model(X_predict)

    last = scaler['Close'].inverse_transform(last[-len(X_predict):].cpu().detach().numpy())

    print(f"Future predicted prices: {last.flatten()}")

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(range(len(future_predictions)), future_predictions, label='Predicted Future Prices', color='green')
plt.title(f'Gold Price Prediction for Next {len(y_test)} Days')
plt.xlabel('Days')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
