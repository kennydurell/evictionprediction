import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

import datetime

from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict
#from arima_models import arimax_by_zip

#importing other models built in this repo
from baseline_model import baseline_model
from arima_models import arimax_by_zip, top_down_estimation_by_zip, arimax_by_month
from random_forest_model import model_random_forest

# Modeling Algorithms
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression

#plotting
import matplotlib.pyplot as plt

df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')

def run_all_models(df_eviction,df_median_housing_price,df_census,df_unemployment):
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    zip_param_dictionary = {'94102': [(2, 1, 0),(1,0,0,6),4],
                '94103': [(0, 1, 1),(1,0,0,6),3],
                 '94105': [(0, 0, 0),(0,0,0,0),4],
                 '94107': [(3, 0, 0),(1,0,0,8),4],
                 '94108': [(0, 1, 1),(1,0,0,6),4],
                 '94109': [(0, 1, 1),(1,0,0,9),4],
                 '94110': [(7, 1, 1),(2,0,0,7),4],
                 '94111': [(0, 0, 0),(1,0,0,6),4],
                 '94112': [(2, 1, 1),(1,0,0,12),4],
                 '94114': [(1, 1, 1),(1,0,0,7),4],
                 '94115': [(1, 1, 1),(2,0,0,3),4],
                 '94116': [(1, 1, 1),(2,0,0,7),4],
                 '94117': [(2, 1, 1),(2,0,0,7),4],
                 '94118': [(1, 1, 1),(2,0,0,7),4],
                 '94121': [(1, 1, 1),(2,0,0,7),4],
                 '94122': [(1, 1, 1),(2,0,0,7),4],
                 '94123': [(0, 1, 1),(1,0,0,12),4],
                 '94124': [(0, 1, 1),(4,0,0,3),4],
                 '94127': [(1, 0, 0),(1,0,0,3),4],
                 '94131': [(1, 0, 0),(1,0,0,3),4],
                 '94132': [(0, 1, 1),(1,0,0,6),3],
                 '94133': [(3, 1, 1),(2,0,0,6),4],
                 '94134': [(3, 1, 1),(2,0,0,6),4],
                 '94158': [(0, 1, 0),(1,0,0,6),4],
                 'Unknown_ZIP': [(4, 1, 1),(0,0,0,0),4]}
    #models
    top_down_by_zip_df = top_down_estimation_by_zip(eviction_median_housing)
    arimax_by_zip_df, rmse = arimax_by_zip(eviction_median_housing, zip_param_dictionary)
    random_forest_df, importance_dict = model_random_forest(eviction_median_housing,10,'auto')


    #linear regression combination of all models
    merged_predictions = linear_regression_combination(arimax_by_zip_df, top_down_by_zip_df, random_forest_df)

    return merged_predictions


def merge_all_models(arimax_by_zip_df, top_down_arimax_df, random_forest_df):
    merged_predictions_1 = pd.merge(arimax_by_zip_df,top_down_arimax_df[['predicted_evictions_by_zip','Month_Year','Address_Zipcode','months_ahead']],\
                            how='inner',left_on=['zip_code','month_year','months_ahead'],right_on= ['Address_Zipcode','Month_Year','months_ahead'],suffixes=('','_top_down_arimax'))
    merged_predictions_2 = pd.merge(merged_predictions_1,random_forest_df,how='inner',\
                                    on= ['zip_code','month_year','months_ahead'], suffixes= ('','_random_forest'))
    months_ahead_prediction_df = merged_predictions_2[merged_predictions_2.months_ahead==3]

    return months_ahead_prediction_df


def build_months_list(months_ahead_prediction_df):
    months = months_ahead_prediction_df[months_ahead_prediction_df>(min(months_ahead_prediction_df['month_year'])+pd.offsets.MonthBegin(3))]['month_year']
    months.dropna(inplace=True)
    months_list = months.drop_duplicates(inplace=False)

    return months_list, months


def linear_regression_combination(arimax_by_zip_df, top_down_arimax_df, random_forest_df):

    months_ahead_prediction_df = merge_all_models(arimax_by_zip_df, top_down_arimax_df,\
                                                                        random_forest_df)

    months_list, months = build_months_list(months_ahead_prediction_df)

    linear_predictions = []
    index_list = []
    rmse_dict={}

    i=0
    for month in months_list:
        #train_test_split
        train = months_ahead_prediction_df[months_ahead_prediction_df['month_year']<(month-pd.offsets.MonthBegin(2))]
        test = months_ahead_prediction_df[months_ahead_prediction_df['month_year']==month]

        X_train, X_test = train[['predicted_evictions','predicted_evictions_by_zip',\
                                                'predicted_evictions_random_forest']],\
                                test[['predicted_evictions','predicted_evictions_by_zip',\
                                                'predicted_evictions_random_forest']]
        y_train,y_test = train['actual_evictions'], test['actual_evictions']

        #fitting model and making predictions, month by month
        lr = LinearRegression()
        model = lr.fit(X_train,y_train)
        predictions= model.predict(X_test)
        linear_predictions.extend(predictions)
        index_list.extend(test.index)
        i+=1
        print i

    #building temporary dataframe to store all the data from the linear regression above
    temp_df = pd.DataFrame(data={'month_year':months,\
                'linear_combination_predicted_evictions':linear_predictions}, index=index_list)

    temp_df['month_year'] = pd.to_datetime(temp_df['month_year'])

    #merging predictions from linear regression with initial prediction datafame
    merged_predictions_3 = pd.merge(months_ahead_prediction_df,temp_df,how='left', left_index=True, right_index=True)

    return merged_predictions_3, 



if __name__ == '__main__':
    merged_predictions  = run_all_models(df_eviction,df_median_housing_price, df_census, df_unemployment)
