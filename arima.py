
from statsmodels.tsa.arima.model import ARIMA
import DataModel
# import pmdarima as pm


if __name__ == '__main__':
    balls, diff = DataModel.load_ssq_blue_diff()
    diff = diff[1:]

    # model = pm.auto_arima(diff, seasonal=False, stepwise=True, trace=True)
    # # 输出最佳的 p, d, q 值
    # print(f"Best ARIMA model: {model.order}")
    # print(f"AIC: {model.aic()}")
    # print(f"BIC: {model.bic()}")

    # train_size = int(len(diff)-10)
    # train_data, test_data = diff[:train_size], diff[train_size:]


    model = ARIMA(balls, order=(11, 0, 11))
    model_fit = model.fit()
    print(model_fit.summary())
    forecast = model_fit.forecast(10)
    print(forecast.values.round().astype(int))

