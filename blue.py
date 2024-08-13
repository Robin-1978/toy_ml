import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义 LSTM 模型
class RandomNumberLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(RandomNumberLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class RandomNumberGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(RandomNumberGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
# 数据准备
def prepare_data(random_numbers, window_size):
    inputs, targets = [], []
    for i in range(len(random_numbers) - window_size):
        inputs.append(random_numbers[i:i + window_size] -1)
        targets.append(random_numbers[i + window_size] -1)  # 将 targets 转换为 0-based index
    return torch.stack(inputs), torch.tensor(targets, dtype=torch.long)


def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.RNN):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    elif isinstance(layer, nn.LSTM):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    elif isinstance(layer, nn.GRU):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

# 生成随机数样本
num_classes = 16
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
random_numbers = torch.tensor(data[:, 8][::-1].astype(int))
# 独热编码
random_numbers_encoded = torch.nn.functional.one_hot(random_numbers - 1, num_classes=num_classes).float()

# 测试不同窗口大小
window_sizes = [3, 6, 12, 24, 36, 48, 60]  # 可以根据需要调整窗口大小
for window_size in window_sizes:
    print(f"\n测试窗口大小: {window_size}")

    # 使用滑动窗口准备数据
    inputs, targets = prepare_data(random_numbers, window_size)  # 使用原始数据生成 targets

    # 转换 inputs 为独热编码
    inputs_encoded = torch.stack([random_numbers_encoded[i:i + window_size] for i in range(len(random_numbers) - window_size)])

    # 创建数据集和数据加载器
    dataset = TensorDataset(inputs_encoded, targets)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = num_classes  # One-hot编码后的长度
    hidden_size = 32
    model = RandomNumberLSTM(input_size, hidden_size, num_classes)
    model.apply(xavier_init)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 1000  # 建议从一个较小的值开始，逐步调整
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = loss_function(output, batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用最后的窗口预测下一个数
    last_sequence = random_numbers_encoded[-window_size:]
    last_sequence = last_sequence.unsqueeze(0)  # 添加批次维度
    model.eval()
    with torch.no_grad():
        output = model(last_sequence)
        predicted_class = torch.argmax(output, dim=1)
        predicted_number = predicted_class.item() + 1  # 转换回1到16

    print(f"预测的下一个随机数是: {predicted_number}")
