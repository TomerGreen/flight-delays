import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


import data_loader


def predict_linear(X,y):
    """
    :param X: original X
    :param y: original y
    :return: prediction of y
    """
    reg = LinearRegression().fit(X, y)
    joblib.dump(reg, 'reg.pkl', compress=0)

    return reg.predict(X)

def calculate_mse(y, y_hat):
    """
    :param y: original y
    :param y_hat: predicted y
    :return: sqrt of MSE
    """
    return math.sqrt(mean_squared_error(y, y_hat))


# X, y_delay, y_factor = data_loader.load_data(path)
# X = np.array(X)
# y_delay = np.array(y_delay)
#
# y_delay_hat = fit_linear(X,y_delay)
# err = calculate_mse(y_delay, y_delay_hat)
#
# print(err)