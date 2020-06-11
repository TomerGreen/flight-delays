import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(path, max_rows=1000000):
    '''

    :param path: path of data file
    :param max_rows: limit to number of loaded rows.
    :return: df with the right features (without the response vector,
             y_delay = column of delay time
             y_factor = column of Delay Factor

    '''
    df = pd.read_csv(path)
    df = df[:max_rows]

    y_delay = df['ArrDelay']
    y_factor = df['DelayFactor']
    df = df.drop(['DelayFactor', 'ArrDelay'], axis=1)

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

    return df, y_delay, y_factor



