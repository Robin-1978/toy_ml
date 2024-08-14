from re import L
import torch
import torch.nn as nn

# 定义参数
num_categories = 33  # 分类号码总数（0 到 32）
embedding_dim = 16  # 嵌入向量的维度
sequence_length = 3  # 每个批次中的序列长度
num_numbers = 6  # 每个序列中的分类号码个数
batch_size = 4  # 批次大小
hidden_size = 32  # 隐藏层大小

# 创建原始数据 (batch_size, sequence_length, num_numbers)
# 假设每个号码都是 0 到 32 之间的整数
class_indices = torch.randint(0, num_categories, (batch_size, sequence_length, num_numbers))
print("Original class indices shape:", class_indices.shape)  # 输出: torch.Size([4, 3, 6])

# 定义嵌入层
embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim)

# 嵌入数据
embedded = embedding(class_indices)
print("Embedded shape:", embedded.shape)  # 输出: torch.Size([4, 3, 6, 16])

# 重新调整形状以适应 LSTM 输入
# 将每个批次的每个时间步的 6 个号码嵌入向量展平为 (batch_size, sequence_length, num_numbers * embedding_dim)
embedded_reshaped = embedded.view(batch_size, sequence_length, num_numbers * embedding_dim)
print("Reshaped embedded shape:", embedded_reshaped.shape)  # 输出: torch.Size([4, 3, 96])

# 定义 LSTM
lstm = nn.LSTM(input_size=num_numbers * embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)

# 输入 LSTM
output, (hn, cn) = lstm(embedded_reshaped)
print("LSTM output shape:", output.shape)  # 输出: torch.Size([4, 3, 32])
print("LSTM hidden state shape:", hn.shape)  # 输出: torch.Size([1, 4, 32])
print("LSTM cell state shape:", cn.shape)    # 输出: torch.Size([1, 4, 32])

# 定义全连接层
fc = nn.Linear(hidden_size, num_categories * num_numbers)

# 将 LSTM 输出转换为分类号码
logits = fc(output[:, -1, :])
logits = logits.view(batch_size,  num_numbers, num_categories,)
print("Logits shape:", logits.shape)  # 输出: torch.Size([4, 6, 33])
