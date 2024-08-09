import torch

# 定义 RNN 分类模型
class RandomNumberClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RandomNumberClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
def he_init(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif isinstance(layer, torch.nn.RNN):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

def xavier_init(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif isinstance(layer, torch.nn.RNN):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

def orthogonal_init(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    elif isinstance(layer, torch.nn.RNN):
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)   


def initialize_model(model, init_func):
    model.apply(init_func)


# 生成随机数样本（假设是 1 到 16）
random_numbers = torch.randint(1, 17, (3000,))

# 转换为独热编码
num_classes = 16
random_numbers_encoded = torch.nn.functional.one_hot(random_numbers - 1, num_classes)

# 随机打乱数据
indices = torch.randperm(len(random_numbers_encoded))
train_indices = indices[:2500]
test_indices = indices[2500:]

train_data = random_numbers_encoded[train_indices]
test_data = random_numbers_encoded[test_indices]

# 数据转换为张量并调整维度
train_data = torch.Tensor(train_data).float().unsqueeze(1)  # 添加序列维度
test_data = torch.Tensor(test_data).float().unsqueeze(1)

# 创建标签
train_labels = torch.argmax(random_numbers_encoded[train_indices], dim=1)
test_labels = torch.argmax(random_numbers_encoded[test_indices], dim=1)

# 模型参数
input_size = 16
hidden_size = 32

# 创建模型
model = RandomNumberClassifier(input_size, hidden_size, num_classes)

initialize_model(model, xavier_init)

# 损失函数和优化器
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    output = model(train_data)
    loss = loss_function(output, train_labels)
    loss.backward()
    optimizer.step()


# 在测试集上进行预测
with torch.no_grad():
    predicted = model(test_data)
    predicted_classes = torch.argmax(predicted, dim=1) + 1  # 转换回原始数字

# 打印预测结果
print("预测值:", predicted_classes)
print("真实值:", random_numbers[test_indices])