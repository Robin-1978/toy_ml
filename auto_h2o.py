import h2o
from h2o.automl import H2OAutoML
import pandas as pd

import numpy as np
import DataModel

def PrepareData(df, features=[], targets=[], window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i + window_size].values)
        y.append(df[targets].iloc[i + window_size])
    return np.array(X), np.array(y), np.expand_dims(np.array(df[features].iloc[-window_size:]), axis=0)


window_size = 3
df = DataModel.load_3d_features()
features=[
    "Ball_1_scale",
    "Ball_2_scale",
    "Ball_3_scale",
    "Ball_1_diff_scale",
    "Ball_2_diff_scale",
    "Ball_3_diff_scale",
    "Ball_1_size",
    "Ball_2_size",
    "Ball_3_size",
    "Ball_1_odd_even",
    "Ball_2_odd_even",
    "Ball_3_odd_even",
    'Ball_1_mean_scale',
    'Ball_2_mean_scale',
    'Ball_3_mean_scale',
    'Ball_1_std_scale',
    'Ball_2_std_scale',
    'Ball_3_std_scale',
]
targets=[
    "Ball_1_odd_even",
]
X, y, PX = PrepareData(df, features=features, targets=targets, window_size=window_size)

X = X.reshape(X.shape[0],-1)
y = y.reshape(-1)

target = 'target'
# 将特征和目标转换为 pandas DataFrame
X_df = pd.DataFrame(X, columns=[f't-{i}' for i in range(window_size * 18, 0, -1)])  # t-1, t-2, ...
y_df = pd.DataFrame(y, columns=[target])


# 合并 X 和 y
df = pd.concat([X_df, y_df], axis=1)

# df['target'] = df['target'].asfactor()

# 启动 H2O 实例
h2o.init()

# 将 pandas DataFrame 转换为 H2O Frame
df_h2o = h2o.H2OFrame(df)

df_h2o[target] = df_h2o[target].asfactor()

# 划分数据集
train, test = df_h2o.split_frame(ratios=[0.8], seed=42)

# 设置目标变量和特征变量
features = [col for col in df.columns if col != target]

# 初始化 H2O AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=42)

# 训练模型
aml.train(x=features, y=target, training_frame=train)

# 评估模型
lb = aml.leaderboard
print(lb)

# 预测
predictions = aml.leader.predict(test)
print(predictions.head())
