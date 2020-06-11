import math
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


from .data_loader import *

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


X, y_delay, y_factor = load_data("../train_data.csv", 200000)


def ridge_test(X, y):
    ridge = Ridge()
    params = {"alpha": [0, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 15, 20, 30, 100, 1e15]}
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    ridge_regressor = GridSearchCV(ridge, params, scoring=mse, cv=2)
    ridge_regressor.fit(X, y)
    print("ridge test")
    print(ridge_regressor.best_params_)
    print(ridge_regressor.best_score_)
    print(ridge_regressor.best_estimator_)

def lasso_test(X, y):
    lasso = Lasso()
    params = {"alpha": [0, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 2, 5, 10, 15, 20, 30, 100, 1e15]}
    mse = make_scorer(mean_squared_error, greater_is_better=False)
    lasso_regressor = GridSearchCV(lasso, params, scoring=mse, cv=2)
    lasso_regressor.fit(X, y)
    print("lasso test")
    print(lasso_regressor.best_params_)
    print(lasso_regressor.best_score_)
    print(lasso_regressor.best_estimator_)


def scalling(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

X, y_delay, y_factor = load_data("203552823/task1/src/flight_data/train_data.csv", 5000)
X = scalling(X)
ridge_test(X, y_delay)
lasso_test(X, y_delay)

# x_train = scaler.fit_transform(x_train)
# x_val = scaler.transform(x_val)
#
#
# print(ridge_test(X, y_delay))
# print(lasso_test(X, y_delay))
