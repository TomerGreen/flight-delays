import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import data_loader


def predict_linear(X,y):
    """
    :param X: original X
    :param y: original y
    :return: prediction of y
    """
    reg = LinearRegression().fit(X, y)
    return reg.predict(X)

def calculate_mse(y, y_hat):
    """
    :param y: original y
    :param y_hat: predicted y
    :return: sqrt of MSE
    """
    return math.sqrt(mean_squared_error(y, y_hat))


X, y_delay, y_factor = data_loader.load_data("../train_data.csv")
# X = np.array(X)
#y_delay = np.array(y_delay)

y_delay_hat = predict_linear(X,y_delay)
err = calculate_mse(y_delay, y_delay_hat)

def ridge_test(X, y):
    ridge = Ridge()
    params = {"alpha":[1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    ridge_regressor = GridSearchCV(ridge, params, scoring='r2', cv=5)
    ridge_regressor.fit(X, y)
    print("ridge test")
    print(ridge_regressor.best_params_)
    print(ridge_regressor.best_score_)

def lasso_test(X, y):
    lasso = Lasso()
    params = {"alpha":[1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_regressor = GridSearchCV(lasso, params, scoring='r2', cv=5)
    lasso_regressor.fit(X, y)
    print("lasso test")
    print(lasso_regressor.best_params_)
    print(lasso_regressor.best_score_)

print("y_delay_hat ", y_delay_hat)
print(err)

