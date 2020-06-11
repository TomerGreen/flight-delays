"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):

===================================================
"""
import numpy as np
import pandas as pd
# from plotnine import *
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib
from .data_loader import *

class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """

        self.class_model = joblib.load('203552823/task1/src/classifier.pkl')
        self.reg_model = joblib.load('203552823/task1/src/reg2.pkl')
        self.path_to_weather = path_to_weather
        self.weather_means_dict = joblib.load('203552823/task1/src/mean_weather_dict.pkl')

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        response = pd.DataFrame()
        weather_data = pd.read_csv(self.path_to_weather)
        X_processed = process_data_test(x, weather_data, self.weather_means_dict)
        ArrDelay = self.reg_model.predict(X_processed)


        response['DelayFactor'] = self.class_model.predict(X_processed)
        response.loc[ArrDelay <= 0, "DelayFactor"] = 'Nan'
        response['ArrDelay'] = ArrDelay

        return response

