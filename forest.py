import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import preprocess


# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = pd.read_csv('./data/ssq/data.csv')

# 获取蓝球号码数据，并逆序排列
blue_ball = data['蓝球'].values[::-1]


print("升降 1:升 0:平 -1:降")
# 提取升降序特征
# 根据相邻两个号码的差值来确定升降序，大于0为升序（1），小于0为降序（-1），等于0为持平（0）
sequence_diff = np.diff(blue_ball)
sequence_trend = np.where(sequence_diff > 0, 1, np.where(sequence_diff < 0, -1, 0))

# 准备数据
X = sequence_trend[:-1].reshape(-1, 1)  # 特征：升降序特征，去掉最后一个作为预测目标
y = sequence_trend[1:]  # 目标：下一个期号的升降序特征

# 划分训练集和测试集
# X_trXain, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
# 创建随机森林分类器模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)
next_sequence_trend = rf_classifier.predict(X[-1].reshape(1, -1))
print("随机森林分类模型Score:", rf_classifier.score(X, y))
print("随机森林分类模型特征重要性:",  rf_classifier.feature_importances_)
print(f"随机森林分类预测的下一个升降序: {next_sequence_trend}")


gb_classifier =  GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X, y)
next_sequence_trend = gb_classifier.predict(X[-1].reshape(1, -1))
print("梯度提升分类模型Score:", gb_classifier.score(X, y))
print("梯度提升分类模型特征重要性:",  gb_classifier.feature_importances_)
print(f"梯度提升分类预测的下一个升降序: {next_sequence_trend}")

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)
next_sequence_trend = rf_regressor.predict(X[-1].reshape(1, -1))
print("随机森林回归模型Score:", rf_regressor.score(X, y))
print("随机森林回归模型征重要性:",rf_regressor.feature_importances_)
print(f"随机森林回归预测的下一个升降序: {next_sequence_trend}")

dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X, y)
next_sequence_trend = dt_regressor.predict(X[-1].reshape(1, -1))
print("决策树回归模型Score:", dt_regressor.score(X, y))
print("决策树回归模型特征重要性:",dt_regressor.feature_importances_)
print(f"决策树回归预测的下一个升降序: {next_sequence_trend}")

####################################################################
#步长
print("步长")
sequence_trend = np.diff(blue_ball)

# 准备数据
X = sequence_trend[:-1].reshape(-1, 1)  # 特征：步长特征，去掉最后一个作为预测目标
y = sequence_trend[1:]  # 目标：下一个期号的步长特征

X ,y = preprocess.ExtractSlidingWindow(blue_ball, 2)
y = y - X[:, -1]
X = np.diff(X, axis=1).reshape(-1, 1)
# X = preprocess.MeanWindow(X).reshape(-1, 1)
# X = preprocess.LinearRegressionWindow(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100,criterion='entropy', random_state=42)

# 训练模型
rf_regressor.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
# 预测
y_pred = rf_classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")

# 示例预测下一个升降序
print("随机森林回归Score:", rf_regressor.score(X_test, y_test))
next_sequence_trend = rf_regressor.predict(X[-1].reshape(1, -1))
print(f"随机森林回归预测的下一个步长: {next_sequence_trend}")

print("随机森林分类Score:", rf_classifier.score(X_test, X_test))
next_sequence_trend = rf_classifier.predict(X[-1].reshape(1, -1))
print(f"随机森林分类预测的下一个步长: {next_sequence_trend}")




# # 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
# data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

# # 获取蓝球号码数据，并逆序排列
# blue_ball = data[:, 8]
# blue_ball = blue_ball[::-1]

# sequence_diff = np.diff(blue_ball)
# sequence_trend = np.where(sequence_diff > 0, 1, np.where(sequence_diff < 0, -1, 0))

# # 准备数据
# X = sequence_trend[:-1].reshape(-1, 1)  # 特征：升降序特征，去掉最后一个作为预测目标
# y = sequence_trend[1:]  # 目标：下一个期号的升降序特征

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # 准备训练数据
# # X_train = blue_ball[:-1].reshape(-1, 1)
# # y_train = blue_ball[1:]

# tree_model = DecisionTreeRegressor(max_depth=10, random_state=0)
# tree_model.fit(X_train, y_train)
# print(f"决策树模型在训练集上的得分: {tree_model.score(X_train, y_train)}")
# # y_pred = tree_model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# print(f"模型准确率: {accuracy}")
# next_sequence_trend = tree_model.predict(X[-1].reshape(1, -1))
# print(f"预测的下一个升降序: {next_sequence_trend}")

# # 创建和训练随机森林模型
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
# rf_model.fit(X_train, y_train)
# print(f"随机森林模型在训练集上的得分: {rf_model.score(X_train, y_train)}")

# gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
# gb_model.fit(X_train, y_train)
# print(f"梯度提升模型在训练集上的得分: {gb_model.score(X_train, y_train)}")

# svm_model = SVR(kernel='rbf')
# svm_model.fit(X_train, y_train)
# print(f"支持向量机模型在训练集上的得分: { svm_model.score(X_train, y_train)}")

# nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=0)
# nn_model.fit(X_train, y_train)
# print(f"神经网络模型在训练集上的得分: {nn_model.score(X_train, y_train)}")

# bagging_model = BaggingRegressor( n_estimators=100, random_state=0)
# bagging_model.fit(X_train, y_train)
# print(f"集成模型在训练集上的得分: {bagging_model.score(X_train, y_train)}")

# adaboost_model = AdaBoostRegressor(n_estimators=100, random_state=0)
# adaboost_model.fit(X_train, y_train)
# print(f"集成模型在训练集上的得分: {adaboost_model.score(X_train, y_train)}")

# tree_model1 = DecisionTreeRegressor(max_depth=10, random_state=0)
# rf_model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
# voting_model = VotingRegressor(estimators=[('dt', tree_model1), ('rf', rf_model1)])
# voting_model.fit(X_train, y_train)
# print(f"集成模型在训练集上的得分: {voting_model.score(X_train, y_train)}")

# # 预测下一个数值
# next_number = blue_ball[-1]
# predicted_tree = tree_model.predict([[next_number]])
# predicted_rf = rf_model.predict([[next_number]])

# print(f"使用决策树模型预测的下一个数值为: {predicted_tree[0]}")
# print(f"使用随机森林模型预测的下一个数值为: {predicted_rf[0]}")


# # 1. 调整决策树模型
# tree_model = DecisionTreeRegressor(random_state=0)
# param_grid_tree = {'max_depth': [None, 10, 20, 30]}
# grid_search_tree = GridSearchCV(tree_model, param_grid_tree, scoring='neg_mean_squared_error', cv=5)
# grid_search_tree.fit(X_train, y_train)
# print("最佳决策树模型参数:", grid_search_tree.best_params_)
# print("最佳决策树模型得分:", grid_search_tree.best_score_)
# print(f"使用决策树模型预测的下一个数值为: {grid_search_tree.predict([[next_number]])}")

# # 2. 调整随机森林模型
# rf_model = RandomForestRegressor(random_state=0)
# param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
# grid_search_rf = GridSearchCV(rf_model, param_grid_rf, scoring='neg_mean_squared_error', cv=5)
# grid_search_rf.fit(X_train, y_train)
# print("最佳随机森林模型参数:", grid_search_rf.best_params_)
# print("最佳随机森林模型得分:", grid_search_rf.best_score_)


# # 3. 调整梯度提升模型
# gb_model = GradientBoostingRegressor(random_state=0)
# param_grid_gb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.05]}
# grid_search_gb = GridSearchCV(gb_model, param_grid_gb, scoring='neg_mean_squared_error', cv=5)
# grid_search_gb.fit(X_train, y_train)
# print("最佳梯度提升模型参数:", grid_search_gb.best_params_)
# print("最佳梯度提升模型得分:", grid_search_gb.best_score_)