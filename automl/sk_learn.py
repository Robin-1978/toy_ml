import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 假设 df 是已经预处理过的时间序列数据
X = df.drop(columns=['target'])
y = df['target']

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 Auto-sklearn 分类器
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, per_run_time_limit=300, seed=42)

# 训练模型
automl.fit(X_train, y_train)

# 评估模型
accuracy = automl.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 获取最佳模型
print(automl.show_models())
