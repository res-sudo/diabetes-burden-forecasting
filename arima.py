import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_predict_arima(train_series, test_series, order=(1, 1, 1)):
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_series))
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    mae = mean_absolute_error(test_series, forecast)
    return forecast, {'RMSE': rmse, 'MAE': mae}