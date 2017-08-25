import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

import datetime

from data_processing import data_processing_eviction, data_processing_housing, merge_data
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict

# Modelling Algorithms

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
# df_eviction = pd.read_csv('/home/ubuntu/eviction_data/Eviction_Notices.csv')
# df_median_housingprice_2 = pd.read_csv('/home/ubuntu/eviction_data/med_sp_zip_code_sf_ca (1).csv')
# df_census = pd.read_csv('/home/ubuntu/eviction_data/ACS_data_total.csv')
# df_unemployment = pd.read_csv('/home/ubuntu/eviction_data/Unemployment_Rate.csv')



df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housingprice_2 = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
#df_buyout = pd.read_csv('Buyout_agreements.csv')


def zip_code_cv(df):
    zip_code_sarima =defaultdict(list)
    i=0
    for zip_code in df['zip_code'].unique():
        results, pdq_best, seasonal_best = param_check(df, zip_code)
        zip_code_sarima[zip_code] = [pdq_best,seasonal_best]
        i+=1
        print i
    return zip_code_sarima


def param_check_overall (df):
    p = range(0,7)
    d= range(0,3)
    q= range(0,7)
    s = [3,6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]
    #(1, 1, 1) (1, 1, 1, 12) 1098.3101292
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


def param_check (df, zip_code):
    p = range(0,7)
    d= range(0,3)
    q= range(0,7)
    s = [3,6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]
    #(1, 1, 1) (1, 1, 1, 12) 1098.3101292
    sorted_df = df[['Month_Year','Eviction_Notice','zip_code']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)

    y = sorted_df[sorted_df['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

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

def sarimax (df):
    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','CASANF0URN']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2.dropna(subset=['CASANF0URN'], inplace=True)
    list_of_zips = sorted_2.zip_code.unique()
    rmse_dict = {}
    i=0
    for zip_code in list_of_zips:
        x = sorted_2[sorted_2['zip_code']==zip_code][[ 'CASANF0URN','Month_Year']].set_index(['Month_Year'], inplace=False)
        y = sorted_2[sorted_2['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
        if len(y)>10 and '2012-01-01' in y.index:
            mod = SARIMAX(endog=y,exog=x,
                                    order=(1,1,1),
                                    seasonal_order=(1,1,1,6),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)


            results = mod.fit()
            pred = results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=False)
            pred_ci = pred.conf_int()
            ax = y['2012':].plot(label='observed')
            pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
            # ax.fill_between(pred_ci.index,
            #         pred_ci.iloc[:, 0],
            #         pred_ci.iloc[:, 1], color='k', alpha=.2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Evictions')
            ax.set_title(zip_code)
            plt.legend()


            y_forecasted = pred.predicted_mean
            y_truth = y['2010-02-01':]
            rmse = (((y_forecasted.values - y_truth.values) ** 2).mean())**.5
            rmse_dict[zip_code]=rmse
            i+=1
            print i

    return rmse_dict


if __name__ == '__main__':
    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_processed = data_processing_housing(df_median_housingprice_2)
    eviction_median_housing = merge_data(df_eviction_processed,df_median_housing_processed, df_census, df_unemployment)
    rmse_dict = sarimax(eviction_median_housing)
    # zip_code_best = zip_code_cv(eviction_median_housing)
    # results, pdq_best, seasonal_best = param_check_overall(eviction_median_housing)
    # print 'Best sarima model for each zip'
    # print zip_code_best
    # print 'Best sarima model for total evictions'
    # print results,pdq_best,seasonal_best
    #
