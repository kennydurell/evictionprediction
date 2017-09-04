import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

import datetime

from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot
#from arima_models import arimax_by_month

from collections import defaultdict
from baseline_model import baseline_model
from random_forest_model import model_random_forest

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
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')





# def zip_code_cv(df):
#     zip_code_arima_best =defaultdict(list)
#     i=0
#     for zip_code in df['Address_Zipcode'].unique():
#         results_list = arimax_by_zip_cv(df, zip_code)
#         zip_code_arima_best[zip_code] = results_list
#         i+=1
#         print i
#     return zip_code_arima_best


def arimax_by_zip_cv (df):

    p = range(0,6)
    d= range(0,2)
    q= range(0,2)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    cv_dict = {}
    baseline_diff = {}
    y_true, y_predict, baseline = baseline_model(df)

    std_list = [1,2,3,4,6,10]

    zip_dict = {'94102': [(5, 1, 0),3],
            '94103': [(0, 1, 1),3],
             '94105': [(0, 0, 0),3],
             '94107': [(0, 0, 0),3],
             '94108': [(0, 1, 1),3],
             '94109': [(0, 1, 1),3],
             '94110': [(1, 1, 1),3],
             '94111': [(0, 0, 0),3],
             '94112': [(2, 1, 1),3],
             '94114': [(0, 1, 1),3],
             '94115': [(1, 1, 1),3],
             '94116': [(0, 1, 1),3],
             '94117': [(2, 1, 1),3],
             '94118': [(0, 1, 1),3],
             '94121': [(0, 1, 1),3],
             '94122': [(0, 1, 1),3],
             '94123': [(0, 1, 1),3],
             '94124': [(0, 1, 1),3],
             '94127': [(0, 0, 0),3],
             '94131': [(0, 0, 0),3],
             '94132': [(0, 1, 1),3],
             '94133': [(0, 1, 1),3],
             '94134': [(1, 1, 1),3],
             '94158': [(0, 1, 0),3],
             'Unknown_ZIP': [(4, 1, 1),3]}
    predictions_df, rmse_final_dict_cv = arimax_by_zip(df, zip_param_dictionary=zip_dict)
    comparison_dict = rmse_final_dict_cv
    zip_dict_test = zip_dict.copy()


    i=0
    for std in std_list:
        for param in pdq:
            zip_dict_test = {x:[param,std] for x in zip_dict_test}
            print zip_dict_test
            break
            predictions_df, rmse_final_dict = arimax_by_zip(df,zip_param_dictionary=zip_dict_test)
            for zip_code in comparison_dict.iterkeys():
                if rmse_final_dict[zip_code]<comparison_dict[zip_code]:
                    comparison_dict[zip_code]=rmse_final_dict[zip_code]
                    baseline_diff[zip_code] = rmse_final_dict[zip_code] - baseline[zip_code]
                    cv_dict[zip_code]=[param,std]

    return comparison_dict, cv_dict, baseline_dff


def arima_by_month_cv (df):
    p = range(0,8)
    d= range(0,3)
    q= range(1,3)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    std_list = [0,1,2,3,5,10]


    param_list = ['param',400000000,3]
    i=0
    for param in pdq:
        for std in std_list:
            predictions_df, rmse = arimax_by_month(df,param,std)

            if rmse<param_list[1]:
                param_list = [param,rmse,std]
            i+=1
            print i
    return param_list


def arima_by_zip_data_transform(df):
    sorted_2 = df.sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day
    sorted_2['CASANF0URN'] = sorted_2['CASANF0URN'].apply(lambda x:-1000 if pd.isnull(x) else x)
    sorted_2 =sorted_2.reset_index(inplace=False)
    # sorted_2.pop('index')

    return sorted_2

def arima_by_zip_months(transformed_df):
    months = transformed_df[transformed_df['Month_Year']>(min(transformed_df['Month_Year'])+pd.offsets.MonthBegin(6))][['Year_S','Month_S','Day_S']]
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


def rf_cv(df):
    predictions_df, rmse_final_dict = model_random_forest(df, num_estimators=10, m_features='auto',std=3)
    y_true, y_predict, baseline = baseline_model(df)
    comparison = rmse_final_dict
    second_dict ={}

    n_estimators = [10,40,80,200,1000]
    max_features =[2,4,'auto']
    std_list=[1,2,3,4]

    for estimator in n_estimators:
        for feature in max_features:
            for std in std_list:
                predictions_df, rmse_final_dict_cv = model_random_forest(df, num_estimators=estimator, m_features = feature, std=std)
                for zip_code in rmse_final_dict_cv.keys():
                    if rmse_final_dict_cv[zip_code] - comparison[zip_code] < 0:
                        baseline_diff = rmse_final_dict_cv[zip_code] - baseline[zip_code]
                        second_dict[zip_code]=(rmse_final_dict_cv[zip_code],baseline_diff,[estimator,feature,std])
                    else:
                        baseline_diff_2 = comparison[zip_code] - baseline[zip_code]
                        second_dict[zip_code] = (comparison[zip_code],baseline_diff_2,[10,'auto',3])
    return second_dict
