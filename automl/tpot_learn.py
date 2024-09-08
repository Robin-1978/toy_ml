from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

# 假设 df 是已经预处理过的时间序列数据
X = df.drop(columns=['target'])
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 TPOT
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

# 训练模型
tpot.fit(X_train, y_train)

# 评估模型
accuracy = tpot.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 导出模型
tpot.export('tpot_best_model.py')
