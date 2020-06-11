import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from sklearn.metrics import accuracy_score


def preprocess_training_data(x, delay, y):
    """
    Filters, splits and scales loaded data to be used in training and evaluation. Also saves the scaler.
    :param x: returned from load_data
    :param delay: returned from load_data
    :param y: returned from load_data
    :return: (x_train, x_val, y_train, y_val)
    """
    # TODO - augment delay data
    # TODO - add option to predict with NoDelay
    x, y = x[delay > 0], y[delay > 0]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=21)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    joblib.dump(scaler, 'classifier_scaler.pkl')
    return x_train, x_val, y_train, y_val


def train_model(x_train, y_train, max_rows=1000000):
    """
    Takes train data and returns a trained classifier.
    """
    x_train, y_train = x_train[:max_rows], y_train[:max_rows]
    classifier = RandomForestClassifier(verbose=2)
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    print(accuracy_score(y_val, y_pred))
    print(pd.crosstab(y_val, y_pred, rownames=['Actual Reasons'], colnames=['Predicted Reasons']))


if __name__ == '__main__':
    print("Loading data")
    x, delay, y = load_data('flight_data/train_data.csv', 'all_weather_data/all_weather_data.csv', max_rows=1000000)
    x_train, x_val, y_train, y_val = preprocess_training_data(x, delay, y)
    print('Fitting model')
    classifier = train_model(x_train, y_train, max_rows=10000000)
    joblib.dump(classifier, 'classifier.pkl')
    evaluate_model(classifier, x_val, y_val)

