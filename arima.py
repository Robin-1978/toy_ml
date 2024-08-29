import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import DataModel
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    print ('start')
    balls, diff = DataModel.load_ssq_blue_diff()
    print ('data loaded')
    model = ARIMA(balls[0:-1], order=(1, 0, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    forecast = model_fit.forecast(1)
    print(forecast.values.round().astype(int))


    # diff2 = diff.diff()
    # model = ARIMA(diff2, order=(5, 0, 0))  
    # model_fit = model.fit()
    # forecast = model_fit.forecast(steps=1)
    # forecast_diff1 = [diff.iloc[-1] + forecast]
    # forecast_values = [balls.iloc[-1] + forecast_diff1]