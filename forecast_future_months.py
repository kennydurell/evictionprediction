import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#datetime imports
from dateutil.relativedelta import relativedelta
import datetime

#imports from other code written for this model
from data_processing_refactor import transform_merge_data
from arima_test_2 import arimax_by_zip, arimax_overall, arima_by_month_cv
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict
from sklearn.metrics import mean_squared_error

# Modelling Algorithms
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housingprice = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')

def forecast_future_evictions(df,months_to_forecast = 3):

    forecast_dict = {}
    params = arima_by_month_cv(df)

    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','Year_S','Month_S','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day
    sorted_2.dropna(subset=['CASANF0URN','CASANF0URN_unemployment_six_months_prior'], inplace=True)
    sorted_2=sorted_2.groupby('Month_Year').sum().reset_index()

    y_train = sorted_2[['Month_Year','Eviction_Notice']]

    for i,row in y_train.Eviction_Notice.iteritems():
        if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*3):
            y_train.Eviction_Notice.iloc[i]=(y_train.Eviction_Notice.mean())

    y_train = y_train.set_index(['Month_Year'], inplace=False)

    mod = ARIMA(endog=y_train,order=params[0])
    results = mod.fit()
    forecast = results.forecast(steps=months_to_forecast)[0]

    for month,f in enumerate(forecast):
        month_of_forecast = max(y_train.index)+ relativedelta(months=month)
        forecast_dict[month_of_forecast]= f

    print 'Month : Forecasted Evictions'

    for month in sorted(forecast_dict.keys()):
        print str(month) + ' : ' + str(forecast_dict[month])

    return forecast_dict


def forecast_future_evictions_by_zip(df,months_to_forecast = 3):
    
    forecast_dict = {}
    params = zip_code_cv(df)



if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housingprice, df_census, df_unemployment)
    forecast_dict = forecast_future_evictions(eviction_median_housing)
