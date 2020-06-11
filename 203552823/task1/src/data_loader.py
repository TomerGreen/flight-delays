from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def skip_convert(val):
    try:
        return float(val)
    except ValueError:
        return


def save_mean_weather_vals(wd, wd_cols):
    """We run this only once"""
    means_dict = dict()
    for colname in wd_cols:
        col = wd[colname]
        mean = col[col != 'None'].astype(float).mean()
        means_dict[colname] = mean
    joblib.dump(means_dict, 'mean_weather_dict.pkl')


def fill_weather_values(wd, wd_cols, mean_weather_dict):
    for colname in wd_cols:
        col = wd[colname]
        col = col.replace('None', str(mean_weather_dict[colname]))
        col = col.fillna(str(mean_weather_dict[colname]))
        wd[colname] = col
    return wd


def augment_weather_data(fd, wd, mean_weather_dict):
    """
    Adds weather data. Drops lines with no weather data.
    :param fd: Flights data table before dummization. Must contain columns: Origin, Dest, Day, Month, Year.
    :param wd: Weather data as given in the csv file.
    :return: The flights data frame with added columns from the weather data and without rows with no weather data.
    """
    # Match wd to fd
    g = wd['day'].str.split('-', expand=True).rename({0: 'Day', 1: 'Month', 2: 'Year'}, axis=1)
    wd['Year'] = g['Year'].astype('uint16') + 2000
    wd['Month'] = g['Month'].astype('uint16')
    wd['Day'] = g['Day'].astype('uint16')
    wd['Origin'] = wd['station'].astype(str)
    wd = wd.drop(['station', 'day'], axis=1)

    # Fix snow data
    wd.loc[(wd['snow_in'] == 'None') | (wd['snow_in'] == '-99') |
           (wd['snow_in'] == '999.9') | (wd['snow_in'] == '999'), 'snow_in'] = '0.0'
    wd.loc[(wd['snowd_in'] == 'None') | (wd['snowd_in'] == '-99'), 'snowd_in'] = '0'

    # Merge and rename columns
    wd_cols = [colname for colname in wd.columns if not colname in ['Origin', 'Dest', 'Day', 'Month', 'Year']]
    origin_cols = ['origin_' + col for col in wd_cols]
    dest_cols = ['dest_' + col for col in wd_cols]
    fd = fd.merge(wd, how='left', on=['Origin', 'Day', 'Month', 'Year'])
    rename_dict = dict(zip(wd_cols, origin_cols))
    fd = fill_weather_values(fd, wd_cols, mean_weather_dict)
    fd = fd.rename(columns=rename_dict)
    wd['Dest'] = wd['Origin']
    wd = wd.drop('Origin', axis=1)
    fd = fd.merge(wd, how='left', on=['Dest', 'Day', 'Month', 'Year'])
    rename_dict = dict(zip(wd_cols, dest_cols))
    fd = fill_weather_values(fd, wd_cols, mean_weather_dict)
    fd = fd.rename(columns=rename_dict)
    return fd


def preprocessing(df, weather_data, mean_weather_dict):
    '''
    :param df:
    :return:
    '''

    df = df.drop(['DayOfWeek'], axis=1).join(pd.get_dummies(df.DayOfWeek, prefix='DayOfWeek_'))

    g = df['FlightDate'].str.split('-', expand=True).rename({0: 'Year', 1: 'Month', 2: 'Day'}, axis=1)
    df = df.drop(['FlightDate'], axis=1)

    # These lines are required for augmenting weather data.
    df['Year'] = g['Year'].astype('uint16')
    df['Month'] = g['Month'].astype('uint16')
    df['Day'] = g['Day'].astype('uint16')
    df['Origin'] = df['Origin'].astype(str)
    df = augment_weather_data(df, weather_data, mean_weather_dict)
    df = df.drop(['Day'], axis=1)

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

    # In order to be able to dropna
    if 'DelayFactor' in df.columns:
        df['DelayFactor'] = df['DelayFactor'].fillna('NoDelay')

    df = df.dropna()

    return df


def load_data(path, weather_path, weather_dict_path, max_rows=1000000):
    '''
    :param path: path of data file
    :param max_rows: limit to number of loaded rows.
    :return: df with the right features (without the response vector,
             y_delay = column of delay time
             y_factor = column of Delay Factor
    '''

    df = pd.read_csv(path)[:max_rows]
    df = preprocessing(df, pd.read_csv(weather_path), joblib.load(weather_dict_path))

    y_delay = df['ArrDelay']
    y_factor = df['DelayFactor']
    df = df.drop(['DelayFactor', 'ArrDelay'], axis=1)

    columns_names = df.columns.tolist()
    joblib.dump(columns_names, 'columns_names.pkl')

    return df, y_delay, y_factor


def load_data_test(path, weather_path, weather_dict_path):
    '''
    :param path:  path of data file
    :return: df with the right features as in the train data
    '''
    columns_names = joblib.load('203552823/task1/src/columns_names.pkl')
    df = preprocessing(pd.read_csv(path), pd.read_csv(weather_path), joblib.load(weather_dict_path))
    df= df.reindex(columns_names, axis=1,fill_value=0)
    return df

def process_data_test(x, weather_data, weather_means_dict):
    """
        :params x: (m,15) dataframe with raw test data
    """
    columns_names = joblib.load('203552823/task1/src/columns_names.pkl')
    df = preprocessing(x, weather_data, weather_means_dict)
    df = df.reindex(columns_names, axis=1, fill_value=0)
    return df
