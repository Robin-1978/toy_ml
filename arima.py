import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import DataModel
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pmdarima as pm


if __name__ == '__main__':
    balls, diff = DataModel.load_ssq_blue_diff()
    diff = diff[1:]

    # model = pm.auto_arima(diff, seasonal=False, stepwise=True, trace=True)
    # # 输出最佳的 p, d, q 值
    # print(f"Best ARIMA model: {model.order}")
    # print(f"AIC: {model.aic()}")
    # print(f"BIC: {model.bic()}")

    train_size = int(len(diff)-10)
    train_data, test_data = diff[:train_size], diff[train_size:]
    model = ARIMA(train_data, order=(5, 0, 0))
    model_fit = model.fit()
    print(model_fit.summary())
    forecast = model_fit.forecast(len(test_data))
    print(forecast)
    mse = mean_squared_error(test_data, forecast)
    print('Mean Squared Error:', mse)
