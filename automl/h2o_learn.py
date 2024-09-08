import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# 启动 H2O 实例
h2o.init()

# 将 pandas DataFrame 转换为 H2O Frame
df_h2o = h2o.H2OFrame(df)

# 划分数据集
train, test = df_h2o.split_frame(ratios=[0.8], seed=42)

# 设置目标变量和特征变量
target = 'target'
features = [col for col in df.columns if col != target]

# 初始化 H2O AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=42)

# 训练模型
aml.fit(x=features, y=target, training_frame=train)

# 评估模型
lb = aml.leaderboard
print(lb)

# 预测
predictions = aml.leader.predict(test)
print(predictions.head())
