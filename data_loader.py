from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def preprocessing(df):
    '''
    :param df:
    :return:
    '''

    df = df.drop(['DayOfWeek'], axis=1).join(pd.get_dummies(df.DayOfWeek, prefix='DayOfWeek_'))

    g = df['FlightDate'].str.split('-', expand=True).rename({0: 'Year', 1: 'Month', 2: 'Day'}, axis=1)
    df = df.drop(['FlightDate'], axis=1)

    df['Year'] = g['Year']
    df['Month'] = g['Month']

    df = df.drop(['Year'], axis=1).join(pd.get_dummies(df.Year, prefix='Year_'))
    df = df.drop(['Month'], axis=1).join(pd.get_dummies(df.Month, prefix='Month_'))

    df = df.drop(['Reporting_Airline'], axis=1).join(pd.get_dummies(df.Reporting_Airline,prefix="Reporting_Airline_"))

    df = df.drop(['Origin'], axis=1).join(pd.get_dummies(df.Origin, prefix='Origin_'))
    df = df.drop(['Dest'], axis=1).join(pd.get_dummies(df.Dest, prefix='Dest_'))

    df['CRSDepTime'] = (df['CRSDepTime'] / 100).round(decimals=0)
    df['CRSArrTime'] = (df['CRSArrTime'] / 100).round(decimals=0)

    df = df.drop(['CRSDepTime'], axis=1).join(pd.get_dummies(df.CRSDepTime, prefix='CRSDepTime_'))
    df = df.drop(['CRSArrTime'], axis=1).join(pd.get_dummies(df.CRSArrTime, prefix='CRSArrTime_'))

    df = df.drop(['Tail_Number', 'Flight_Number_Reporting_Airline', 'OriginCityName', 'OriginState',
                  'DestCityName', 'DestState'], axis=1)

    return df

def load_data(path,max_rows=500000):
    '''
    :param path: path of data file
    :param max_rows: limit to number of loaded rows.
    :return: df with the right features (without the response vector,
             y_delay = column of delay time
             y_factor = column of Delay Factor
    '''

    df = preprocessing(pd.read_csv(path))
    df = df[:max_rows]

    y_delay = df['ArrDelay']
    y_factor = df['DelayFactor']
    df = df.drop(['DelayFactor', 'ArrDelay'], axis=1)

    columns_names = df.columns.tolist()
    joblib.dump(columns_names, 'columns_names.pkl')

    return df, y_delay, y_factor



def load_data_test(path):
    '''
    :param path:  path of data file
    :return: df with the right features as in the train data
    '''
    columns_names = joblib.load('columns_names.pkl')
    df = preprocessing(pd.read_csv(path))
    df= df.reindex(columns_names, axis=1,fill_value=0)
    return df

