import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from sklearn.metrics import accuracy_score


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


def preprocess_data(x, y):
    pass


def train_model(x_train, y_train, max_rows=1000000):
    """
    Takes train data and returns a trained classifier.
    """
    x_train, y_train = x_train[:max_rows], y_train[:max_rows]
    classifier = RandomForestClassifier(verbose=2)
    print('Fitting model')
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    print(accuracy_score(y, y_pred))
    print(pd.crosstab(y_val, y_pred, rownames=['Actual Reasons'], colnames=['Predicted Reasons']))


if __name__ == '__main__':
    x, delay, y = load_data('flight_data/train_data.csv')
    print("Data loaded")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=21)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    classifier = train_model(x, y, max_rows=100000000)
    joblib.dump(classifier, 'classifier.pkl')
    evaluate_model(classifier, x_val, y_val)

