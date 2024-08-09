import numpy as np
from hmmlearn import hmm

# 设置随机种子，保证结果的可复现性
# np.random.seed(0)

# 获取蓝球号码数据，并逆序排列
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
blue_ball = data[:, 8][::-1].astype(int)

# sequence_diff = np.diff(blue_ball)
# sequence_trend = np.where(sequence_diff > 0, 2, np.where(sequence_diff < 0, 0, 1))

# # 准备数据
# X = sequence_trend.reshape(-1, 1)  # 特征：升降序特征
# # y = sequence_trend[1:]  # 目标：下一个期号的升降序特征

# xxx = X[-1]

# 准备观测序列
# observations = blue_ball.reshape(-1, 1)  # 将蓝球号码整理成观测序列的格式

# 定义并训练HMM模型
model = hmm.MultinomialHMM(n_components=16, random_state=42, n_iter=100)  # 假设有16个隐藏状态
model.startprob_ = np.ones(16) / 16
model.fit(blue_ball.reshape(-1, 1))

# 预测下一个状态（蓝球号码）
next_observation = np.array(blue_ball[-1].reshape(-1, 1))  # 最后一个观测值作为输入预测下一个
predicted_state = model.predict(next_observation)[0]

print(f"预测的下一个状态（蓝球号码）: {next_observation} -> {predicted_state + 1}")  # 加1是因为预测值是0到15的索引