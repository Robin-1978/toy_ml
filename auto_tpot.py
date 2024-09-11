from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import DataModel

def PrepareData(df, features=[], targets=[], window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i + window_size].values)
        y.append(df[targets].iloc[i + window_size])
    return np.array(X), np.array(y), np.expand_dims(np.array(df[features].iloc[-window_size:]), axis=0)


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
X, y, PX = PrepareData(df, features=features, targets=targets, window_size=3)
X = X.reshape(X.shape[0],-1)
y = y.reshape(-1)
# y = y - 1
split_index =int(len(X) * 0.95)
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]


# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 TPOT
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

# 训练模型
tpot.fit(X_train, y_train)

# 评估模型
accuracy = tpot.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 导出模型
tpot.export('tpot_best_model.py')
