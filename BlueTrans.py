import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, nhead=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, 1)
        self.transformer = nn.Transformer(
            hidden_size, nhead, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        print("Before Embedding: ", X.shape)
        embedded = self.embedding(X)
        print("After Embedding: ", embedded.shape)
        embedded.squeeze(2)
        # transformed = self.transformer(embedded.permute(0, 1, 3, 2))
        reshaped = embedded.reshape(embedded.shape[0], embedded.shape[1], embedded.shape[2])
        tgt = torch.zeros_like(X)
        transformed = self.transformer(reshaped, tgt)
        pooled = torch.mean(transformed, dim=1)
        output = self.fc(pooled)
        return output

def prepare_data(random_numbers, window_size):
    inputs, targets = [], []
    for i in range(len(random_numbers) - window_size):
        inputs.append(random_numbers[i:i + window_size] -1)
        targets.append(random_numbers[i + window_size] -1)  # 将 targets 转换为 0-based index
    return torch.stack(inputs), torch.tensor(targets, dtype=torch.long)

# 生成随机数样本
num_classes = 16
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
random_numbers = torch.tensor(data[:, 8][::-1].astype(int))
# 独热编码
random_numbers_encoded = torch.nn.functional.one_hot(random_numbers - 1, num_classes=num_classes).long()

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
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = num_classes  # One-hot编码后的长度
    hidden_size = 32
    model = TransformerModel(input_size, hidden_size, num_classes)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10  # 建议从一个较小的值开始，逐步调整
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
