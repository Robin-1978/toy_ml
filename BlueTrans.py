from enum import KEEP
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math


class TransformerModel(nn.Module):
    def __init__(self, num_class=16, embedding_size=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_class, embedding_size)
        self.transformer = nn.Transformer(
            d_model=num_class * embedding_size, batch_first=True
        )
        self.fc = nn.Linear(num_class * embedding_size, num_class)

    def forward(self, src, tgt):
        src = self.embedding(src).reshape(src.shape[0], src.shape[1], -1)
        tgt = self.embedding(tgt).reshape(tgt.shape[0], tgt.shape[1], -1)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output


def prepare_data(data_sequence, num_classes, input_seq_len, output_seq_len):
    data_sequence = data_sequence - 1
    num_samples = len(data_sequence) - input_seq_len - output_seq_len + 1
    src = torch.tensor( np.array(
        [data_sequence[i : i + input_seq_len] for i in range(num_samples)]
    ))
    tgt = torch.tensor(np.array(
        [
            data_sequence[i + input_seq_len : i + input_seq_len + output_seq_len]
            for i in range(num_samples)
        ]
    ))
    # src = torch.clamp(src, 0, num_classes - 1)
    src = torch.nn.functional.one_hot(src.to(torch.int64), num_classes=num_classes)
    tgt = torch.nn.functional.one_hot(tgt.to(torch.int64), num_classes=num_classes)
    return src, tgt
    inputs, targets = [], []

    for i in range(len(random_numbers) - window_size):
        inputs.append(random_numbers[i : i + window_size] - 1)
        targets.append(
            random_numbers[i + window_size] - 1
        )  # 将 targets 转换为 0-based index
    return torch.stack(inputs), torch.stack(targets)


# 生成随机数样本
num_classes = 16
data = np.loadtxt("./data/ssq/data.csv", delimiter=",", skiprows=1)
random_numbers = data[:, 8][::-1].astype(int)
if random_numbers.min() < 1 or random_numbers.max() > num_classes:
    raise ValueError(
        "Values in random_numbers are out of the expected range (1 to 16)."
    )

# 测试不同窗口大小
window_sizes = [3, 6, 12, 24, 36, 48, 60]  # 可以根据需要调整窗口大小
for window_size in window_sizes:
    predict_size = 1
    print(f"\n测试窗口大小: {window_size}, 预测长度: {predict_size}")

    # 使用滑动窗口准备数据
    inputs, targets = prepare_data(
        random_numbers , num_classes, window_size, predict_size
    )  # 使用原始数据生成 targets
    # targets = targets.unsqueeze(1)
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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    num_epochs = 100  # 建议从一个较小的值开始，逐步调整
    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_inputs, batch_targets)
            loss = loss_function(output.reshape(output.shape[0], -1), torch.argmax(batch_targets.reshape(batch_targets.shape[0], -1), dim=1))
            loss.backward()
            optimizer.step()

        # if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 使用最后的窗口预测下一个数
    # last_sequence = inputs[-1].unsqueeze(0)

    # last_sequence = last_sequence.unsqueeze(0)  # 添加批次维度
    last_sequence = torch.tensor(random_numbers[-window_size:]).unsqueeze(0)-1
    last_sequence = torch.nn.functional.one_hot(last_sequence.to(torch.int64), num_classes=num_classes)

    last_tgt = torch.zeros((1, predict_size, num_classes), dtype=torch.long)
    last_number = []
    model.eval()
    with torch.no_grad():
        for i in range(predict_size):
            output = model(last_sequence, last_tgt[: i + 1])
            predicted_class = torch.argmax(output[-1], dim=1)
            last_number.append(predicted_class + 1)
            last_tgt[:, i] = torch.nn.functional.one_hot(predicted_class, num_classes=num_classes)

    print(f"预测的下一个随机数是: {last_number}")
