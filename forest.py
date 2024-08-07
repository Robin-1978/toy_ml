import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

# 获取蓝球号码数据，并逆序排列
blue_ball = data[:, 8]
blue_ball = blue_ball[::-1]

# 准备训练数据
X_train = blue_ball[:-1].reshape(-1, 1)
y_train = blue_ball[1:]

tree_model = DecisionTreeRegressor(max_depth=10, random_state=0)
tree_model.fit(X_train, y_train)
print(f"决策树模型在训练集上的得分: {tree_model.score(X_train, y_train)}")

# 创建和训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
rf_model.fit(X_train, y_train)
print(f"随机森林模型在训练集上的得分: {rf_model.score(X_train, y_train)}")

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
gb_model.fit(X_train, y_train)
print(f"梯度提升模型在训练集上的得分: {gb_model.score(X_train, y_train)}")

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
print(f"支持向量机模型在训练集上的得分: { svm_model.score(X_train, y_train)}")

nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=0)
nn_model.fit(X_train, y_train)
print(f"神经网络模型在训练集上的得分: {nn_model.score(X_train, y_train)}")

bagging_model = BaggingRegressor( n_estimators=100, random_state=0)
bagging_model.fit(X_train, y_train)
print(f"集成模型在训练集上的得分: {bagging_model.score(X_train, y_train)}")

adaboost_model = AdaBoostRegressor(n_estimators=100, random_state=0)
adaboost_model.fit(X_train, y_train)
print(f"集成模型在训练集上的得分: {adaboost_model.score(X_train, y_train)}")

tree_model1 = DecisionTreeRegressor(max_depth=10, random_state=0)
rf_model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
voting_model = VotingRegressor(estimators=[('dt', tree_model1), ('rf', rf_model1)])
voting_model.fit(X_train, y_train)
print(f"集成模型在训练集上的得分: {voting_model.score(X_train, y_train)}")

# 预测下一个数值
next_number = blue_ball[-1]
predicted_tree = tree_model.predict([[next_number]])
predicted_rf = rf_model.predict([[next_number]])

print(f"使用决策树模型预测的下一个数值为: {predicted_tree[0]}")
print(f"使用随机森林模型预测的下一个数值为: {predicted_rf[0]}")


# 1. 调整决策树模型
tree_model = DecisionTreeRegressor(random_state=0)
param_grid_tree = {'max_depth': [None, 10, 20, 30]}
grid_search_tree = GridSearchCV(tree_model, param_grid_tree, scoring='neg_mean_squared_error', cv=5)
grid_search_tree.fit(X_train, y_train)
print("最佳决策树模型参数:", grid_search_tree.best_params_)
print("最佳决策树模型得分:", grid_search_tree.best_score_)
print(f"使用决策树模型预测的下一个数值为: {grid_search_tree.predict([[next_number]])}")

# 2. 调整随机森林模型
rf_model = RandomForestRegressor(random_state=0)
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, scoring='neg_mean_squared_error', cv=5)
grid_search_rf.fit(X_train, y_train)
print("最佳随机森林模型参数:", grid_search_rf.best_params_)
print("最佳随机森林模型得分:", grid_search_rf.best_score_)


# 3. 调整梯度提升模型
gb_model = GradientBoostingRegressor(random_state=0)
param_grid_gb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.05]}
grid_search_gb = GridSearchCV(gb_model, param_grid_gb, scoring='neg_mean_squared_error', cv=5)
grid_search_gb.fit(X_train, y_train)
print("最佳梯度提升模型参数:", grid_search_gb.best_params_)
print("最佳梯度提升模型得分:", grid_search_gb.best_score_)