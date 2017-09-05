import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyflux as pf
import logging
import datetime

#imports from other code in this repo
from data_processing_refactor import transform_merge_data
from cross_validation import arima_by_zip_data_transform, arima_by_zip_months
from pandas.tools.plotting import autocorrelation_plot
from cross_validation import arima_by_zip_data_transform
from random_forest_model import random_forest_model_data_cleaning

from collections import defaultdict

# Modelling Algorithms
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


#importing files
df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')
df_future_data = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/test_data.csv')


def run_random_forest_forecast(df_eviction,df_median_housing_price, df_census, df_unemployment,df_future_data,months_ahead=3):
    '''Runs data_processing for all data and subsequently the random forest model forecast functions. See below for detailed doc strings within each'''

    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    random_forest_forecast_df = random_forest_forecast(eviction_median_housing,months_ahead,df_future_data)

    return random_forest_forecast_df



def random_forest_forecast(df, months_ahead,df_future_data):
    """
    Forecasts specified months_ahead based on past eviction data for San Francisco.

    Parameters:
    df - eviction dataframe, processed and combined with census, unemployment data etc.
    months_ahead - number of months ahead to run the prediction.
    df_future_data - exogenous variables from future datapoints,
    namely unemployment and capital improvements in previous years.

    Output:
    predictions_df - dataframe with predictions of eviction notices, by zip, for each of the future months.
    """

    merged_sorted = random_forest_model_data_cleaning(df)
    X_test = random_forest_model_data_cleaning(future_data_processing(df_future_data))

    #importance_dict={}

    now = datetime.datetime.now()
    this_month = pd.to_datetime(str(now.year)+ '-'+ str(now.month) + '-'+str(1))
    X_test=X_test[(X_test.Month_Year>this_month)&(X_test.Month_Year<(this_month+pd.offsets.MonthBegin(months_ahead+1)))].sort_values('Month_Year')
    months_list = X_test.Month_Year.tolist()
    zips = X_test.Address_Zipcode.tolist()

    y_train = merged_sorted.pop('Eviction_Notice')
    X_train = merged_sorted.drop(['Month_Year','Month_S','Year_S','Day_S'], axis=1)
    X_test = X_test.drop(['Month_Year','Month_S','Year_S','Day_S','Eviction_Notice'], axis=1)


    for zip_code in X_train['Address_Zipcode'].unique():
        indices_2 = X_train[X_train.Address_Zipcode==zip_code].index
        for i,value in y_train.iloc[indices_2].iteritems():
            if value>(y_train.iloc[indices_2].mean()+ y_train.iloc[indices_2].std()*3):
                y_train.iloc[i]=y_train.iloc[indices_2].mean()

    X_train = X_train.drop('Address_Zipcode',axis=1)
    X_test = X_test.drop('Address_Zipcode',axis=1)

    predictions_df = random_forest_forecast_fit_predict(X_train,y_train,X_test,zips,months_list)

    return predictions_df





def random_forest_forecast_fit_predict(X_train,y_train,X_test,zips,months_list):

    """
    Performs fit and forecast for SF eviction data.

    Parameters:
    X_train - previous eviction data
    y_train - previous evictions, with zip dummified and included as exogenous variables
    X_test - exogenous, time-lagged variables from future months
    zips - list of zips to append back onto the data after it is fit
    months_list - list of months to append back onto the data after it is fit

    Output:
    predictions_df - dataframe with predictions of eviction notices, by zip, for each of the future months.
    """
    rfr = RandomForestRegressor(n_estimators = 100, max_features='auto')
    rfr.fit(X_train,y_train)
    y_hat = rfr.predict(X_test).tolist()
    predictions_df = pd.DataFrame(data={'predicted_evictions':y_hat,\
                        'zip_code': zips, 'month_year':months_list})

    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])
    return predictions_df

def future_data_processing(df):
    '''Basic processing/transformation of future exogenous variables to align them with the format of the past eviction data.'''
    future_df = transform_merge_data(df,df_median_housing_price, df_census, df_unemployment)
    return future_df



if __name__=='__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    random_forest_forecast_df = random_forest_forecast(eviction_median_housing,3,df_future_data)
    #eviction = run_random_forest_forecast(df_eviction,df_median_housing_price, df_census, df_unemployment,df_future_data,months_ahead=3)
