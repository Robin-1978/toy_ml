import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

    
class TransformerModel(nn.Module):
    def __init__(self, num_class = 16, d_model=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_class, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=8, batch_first=True)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.fc = nn.Linear(d_model, num_class)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        print(f"{src.shape}, {tgt.shape}")
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

def prepare_data(random_numbers, window_size):
    inputs, targets = [], []
    for i in range(len(random_numbers) - window_size):
        inputs.append(random_numbers[i:i + window_size] -1)
        targets.append(random_numbers[i + window_size] -1)  # 将 targets 转换为 0-based index
    return torch.stack(inputs), torch.stack(targets)

# 生成随机数样本
num_classes = 16
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
random_numbers = torch.tensor(data[:, 8][::-1].astype(int), dtype=torch.long)
if random_numbers.min() < 1 or random_numbers.max() > num_classes:
    raise ValueError("Values in random_numbers are out of the expected range (1 to 16).")

# 测试不同窗口大小
window_sizes = [3, 6, 12, 24, 36, 48, 60]  # 可以根据需要调整窗口大小
for window_size in window_sizes:
    print(f"\n测试窗口大小: {window_size}")

    # 使用滑动窗口准备数据
    inputs, targets = prepare_data(random_numbers, window_size)  # 使用原始数据生成 targets
    targets = targets.unsqueeze(1)
    # 创建数据集和数据加载器
    dataset = TensorDataset(inputs, targets)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = num_classes  # One-hot编码后的长度
    model = TransformerModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10  # 建议从一个较小的值开始，逐步调整
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_inputs, batch_targets)
            loss = loss_function(output.view(-1, output.size(-1)), batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 使用最后的窗口预测下一个数
    last_sequence = inputs[-window_size:]
    last_sequence = last_sequence.unsqueeze(0)  # 添加批次维度
    last_tgt = [0]
    last_tgt.unsqueeze(0)
    np.zeros_like(last_tgt)
    model.eval()
    with torch.no_grad():
        output = model(last_sequence, last_tgt)
        predicted_class = torch.argmax(output, dim=1)
        predicted_number = predicted_class.item() +1 # 转换回1到16

    print(f"预测的下一个随机数是: {predicted_number}")
