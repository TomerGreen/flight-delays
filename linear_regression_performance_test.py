import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import data_loader

path = "train_data.csv"

def scalling(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
    # y = scaler.transform(y)

def predict_linear(X, y):
    """
    :param X: original X
    :param y: original y
    :return: prediction of y
    """
    reg = LinearRegression()
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    # linear_regressor = GridSearchCV(reg, scoring=mse, cv=2)
    # print("linear test")
    # print(linear_regressor.best_params_)
    # print(linear_regressor.best_score_)
    # print(linear_regressor.best_estimator_)

def calculate_mse(y, y_hat):
    """
    :param y: original y
    :param y_hat: predicted y
    :return: sqrt of MSE
    """
    return math.sqrt(mean_squared_error(y, y_hat))

def ridge_test(X, y):
    ridge = Ridge()
    params = {"alpha":[0, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 15, 20, 30, 100, 1e15]}
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    ridge_regressor = GridSearchCV(ridge, params, scoring=mse, cv=2)
    ridge_regressor.fit(X, y)
    print("ridge test")
    print(ridge_regressor.best_score_)
    print(ridge_regressor.best_estimator_)

def lasso_test(X, y):
    lasso = Lasso()
    params = {"alpha":[0, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 15, 20, 30, 100, 1e15]}
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    lasso_regressor = GridSearchCV(lasso, params, scoring=mse, cv=2)
    lasso_regressor.fit(X, y)
    print("lasso test")
    print(lasso_regressor.best_score_)
    print(lasso_regressor.best_estimator_)

def lasso_fit(X, y):
    lasso = Lasso(alpha=20).fit(X,y)
    y_hat = lasso.predict(X)
    print(calculate_mse(y, y_hat))



X, y_delay, y_factor = data_loader.load_data(path, 5000)
X = np.array(scalling(X))

# y_delay_hat = predict_linear(X,y_delay)
# err = calculate_mse(y_delay, y_delay_hat)
# print(err)
lasso_fit(X, y_delay)
# ridge_test(X,y_delay)
lasso_test(X,y_delay)

