import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

import datetime

from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict

# Modelling Algorithms
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

#For use on EC2 instances
# df_eviction = pd.read_csv('/home/ubuntu/eviction_data/Eviction_Notices.csv')
# df_median_housingprice_2 = pd.read_csv('/home/ubuntu/eviction_data/med_sp_zip_code_sf_ca (1).csv')
# df_census = pd.read_csv('/home/ubuntu/eviction_data/ACS_data_total.csv')
# df_unemployment = pd.read_csv('/home/ubuntu/eviction_data/Unemployment_Rate.csv')
#

#importing files
df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')






def zip_code_cv(df):
    zip_code_arima_best =defaultdict(list)
    i=0
    for zip_code in df['Address_Zipcode'].unique():
        results_list = arimax_by_zip_cv(df, zip_code)
        zip_code_arima_best[zip_code] = results_list
        i+=1
        print i
    return zip_code_arima_best


def arimax_by_zip_cv (df,zip_code):

    p = range(0,6)
    d= range(0,2)
    q= range(0,2)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    sorted_df = df[['Month_Year','Eviction_Notice','Address_Zipcode']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)

    aic_list = ['param',400000000]
    i=0

    y_train = sorted_df[sorted_df['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    for param in pdq:
        try:
            mod = ARIMA(endog=y_train,order=param)
            results = mod.fit()
            if results.aic<aic_list[1]:
                aic_list=[param,results.aic]
            i+=1
            print i
        except:
            continue

    return aic_list

def arima_by_month_cv (df):
    p = range(0,8)
    d= range(0,3)
    q= range(1,3)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    sorted_df = df[['Month_Year','Eviction_Notice']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)
    y_train=sorted_df.set_index(['Month_Year'], inplace=False)

    aic_list = ['param',400000000]
    i=0
    for param in pdq:
        try:
            mod = ARIMA(endog=y_train,order=param)
            results = mod.fit()
            if results.aic<aic_list[1]:
                aic_list=[param,results.aic]
            i+=1
            print i
        except:
            continue
    return aic_list


def arima_by_zip_data_transform(df):
    sorted_2 = df.sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day
    #sorted_2.dropna(subset=['CASANF0URN','percent_white_population_previous_year'], inplace=True)
    sorted_2.reset_index(inplace=True)

    return sorted_2

def arima_by_zip_months(transformed_df):
    months = transformed_df[transformed_df['Month_Year']>(min(transformed_df['Month_Year'])+pd.offsets.MonthBegin(3))][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]

    return months_list


def sarimax_cv (df):
    p = range(0,4)
    d= range(0,2)
    q= range(0,2)
    s = [6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

    #reformatting the data for use in SARIMAX model
    sorted_df = df[['Month_Year','Eviction_Notice']].groupby('Month_Year').sum().reset_index()
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)
    y = sorted_df[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    results = 40000000
    pdq_best = None
    seasonal_best = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(y,order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                if results.aic<results:
                    results = results.aic
                    pdq_best = param
                    seasonal_best = param_seasonal
            except:
                continue
    return results, pdq_best, seasonal_best


def sarimax_by_zip_cv (df, zip_code):
    p = range(0,4)
    d= range(0,2)
    q= range(0,2)
    s = [6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

    #reformatting the data for use in SARIMAX model
    sorted_df = df[['Month_Year','Eviction_Notice','Address_Zipcode']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)

    y = sorted_df[sorted_df['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    results = 40000000
    pdq_best = None
    seasonal_best = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(y,order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                if results.aic<results:
                    results = results.aic
                    pdq_best = param
                    seasonal_best = param_seasonal
            except:
                continue
    return results, pdq_best, seasonal_best
