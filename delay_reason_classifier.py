import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_hour(time_str):
    return int(time_str/100)


def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data[['DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'Distance', 'CRSDepTime', 'ArrDelay',
                 'DelayFactor']]
    data['CRSDepTime'] = data['CRSDepTime'].apply(parse_hour)
    data = data[data['ArrDelay'] > 0]
    x = data.drop(columns=['DelayFactor'])
    y = data['DelayFactor']
    return x, y


def train_model(x, y):
    y, uniques = pd.factorize(y)
    classifier = RandomForestClassifier()
    classifier.fit(x, y)
    return classifier


if __name__ == '__main__':
    x, y = load_data('flight_data/train_data.csv')
    train_model(x, y)
    print(x.shape)
