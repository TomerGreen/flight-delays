import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def parse_hour(time_str):
    return int(time_str/100)


# def load_data(data_path):
#     data = pd.read_csv(data_path)
#     data = data[['DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'Distance', 'CRSDepTime', 'ArrDelay',
#                  'DelayFactor']]
#     data['CRSDepTime'] = data['CRSDepTime'].apply(parse_hour)
#     data = data[data['ArrDelay'] > 0]
#     x = data.drop(columns=['DelayFactor'])
#     y = data['DelayFactor']
#     return x, y


def preprocess_training_data(x, delay, y):
    """
    Filters, splits and scales loaded data to be used in training and evaluation. Also saves the scaler.
    :param x: returned from load_data
    :param delay: returned from load_data
    :param y: returned from load_data
    :return: (x_train, x_val, y_train, y_val)
    """
    x, y = x[delay > 0], y[delay > 0]
    print("Data loaded")
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
    print('Fitting model')
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    print(accuracy_score(y_val, y_pred))
    print(pd.crosstab(y_val, y_pred, rownames=['Actual Reasons'], colnames=['Predicted Reasons']))

def get_bestF(x_train, y_train, max_rows = int(1e5)):
    x_train, y_train = x_train[:max_rows], y_train[:max_rows]
    classifier = RandomForestClassifier(verbose=2)

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf)

    gridF = GridSearchCV(classifier, hyperF, cv=3, verbose=1,
                         n_jobs=-1)
    bestF = gridF.fit(x_train, y_train)
    return bestF

if __name__ == '__main__':
    x, delay, y = load_data('flight_data/train_data.csv')
    x_train, x_val, y_train, y_val = preprocess_training_data(x, delay, y)
    # classifier = train_model(x_train, y_train, max_rows=int(1e3))
    bestF = get_bestF(x_train, y_train, max_rows=int(1e3))
    classifier = bestF.best_estimator_
    joblib.dump(classifier, 'classifierBestF1.pkl',compress=9)
    evaluate_model(classifier, x_val, y_val)

