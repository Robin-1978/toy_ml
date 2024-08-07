import numpy as np
from hmmlearn import hmm

# 设置随机种子，保证结果的可复现性
np.random.seed(0)

# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

# 获取蓝球号码数据，并逆序排列
blue_ball = data[:, 8]
blue_ball = blue_ball[::-1]
blue_ball = blue_ball.astype(int)
# 准备观测序列
observations = blue_ball.reshape(-1, 1)  # 将蓝球号码整理成观测序列的格式

# 定义并训练HMM模型
model = hmm.MultinomialHMM(n_components=16)  # 假设有16个隐藏状态
model.fit(observations)

# 预测下一个状态（蓝球号码）
next_observation = np.array([[blue_ball[-1]]])  # 最后一个观测值作为输入预测下一个
predicted_state = model.predict(next_observation)[0]

print(f"预测的下一个状态（蓝球号码）: {next_observation} -> {predicted_state + 1}")  # 加1是因为预测值是0到15的索引