import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# read from csv file path: ./data/ssq/data.csv , delimiter: ',' include header
table = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

#format : id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
# Get the blue ball numbers as input
blue_ball = table[:,8]
blue_ball = blue_ball[::-1]

print (blue_ball)

# # 尝试不同的参数组合
# for p in range(3):
#     for d in range(2):
#         for q in range(3):
#             order = (p, d, q)
#             try:
#                 model = ARIMA(blue_ball, order=order)
#                 model_fit = model.fit()
#                 if model_fit.aic < best_aic:
#                     best_aic = model_fit.aic
#                     best_order = order
#             except:
#                 continue

# print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")

# 拟合ARIMA模型
model = ARIMA(blue_ball, order=(0, 0, 0))
model_fit = model.fit()
print(model_fit.summary())

predicted_value = model_fit.forecast(steps=1)[0]
predicted_value = int(predicted_value)
print(predicted_value)