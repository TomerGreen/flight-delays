import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def augment_weather_data(flights_data, weather_data_path):
    wd = pd.read_csv(weather_data_path)
    g = wd['day'].str.split('-', expand=True).rename({0: 'Day', 1: 'Month', 2: 'Year'}, axis=1)
    wd['Year'] = g['Year'].astype('uint16') + 2000
    wd['Month'] = g['Month'].astype('uint16')
    wd['Day'] = g['Day'].astype('uint16')
    wd['Origin'] = wd['station'].astype(str)
    wd = wd.drop(['station', 'day'], axis=1)
    wd.loc[(wd['snow_in'] == 'None') | (wd['snow_in'] == '-99') |
           (wd['snow_in'] == '999.9') | (wd['snow_in'] == '999'), 'snow_in'] = '0.0'
    wd.loc[(wd['snowd_in'] == 'None'), 'snowd_in'] = '0'
    wd_cols = [colname for colname in wd.columns if not colname in ['Origin', 'Dest', 'Day', 'Month', 'Year']]
    origin_cols = ['origin_' + col for col in wd_cols]
    dest_cols = ['dest_' + col for col in wd_cols]
    flights_data = flights_data.merge(wd, how='left', on=['Origin', 'Day', 'Month', 'Year'])
    rename_dict = dict(zip(wd_cols, origin_cols))
    flights_data = flights_data.rename(columns=rename_dict)
    wd['Dest'] = wd['Origin']
    wd = wd.drop('Origin', axis=1)
    flights_data = flights_data.merge(wd, how='left', on=['Dest', 'Day', 'Month', 'Year'])
    rename_dict = dict(zip(wd_cols, dest_cols))
    flights_data = flights_data.rename(columns=rename_dict)
    flights_data = flights_data.dropna()
    return flights_data


def load_data(path, weather_path, max_rows=1000000):
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

    df['Year'] = g['Year'].astype('uint16')
    df['Month'] = g['Month'].astype('uint16')
    df['Day'] = g['Day'].astype('uint16')

    df['Origin'] = df['Origin'].astype(str)
    df = augment_weather_data(df, weather_path)

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

