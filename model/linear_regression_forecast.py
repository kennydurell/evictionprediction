import warnings
import itertools
import pandas as pd
import numpy as np


import datetime

from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict
#from arima_models import arimax_by_zip

#importing other models built in this repo
from baseline_model import baseline_model
from arima_models import arimax_by_zip, top_down_estimation_by_zip, arimax_by_month
from random_forest_model import model_random_forest
from top_down_forecast import run_top_down_forecast
from random_forest_forecast import run_random_forest_forecast
from plotting import plot_by_zips

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
df_future_data = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/test_data.csv')


def predict_evictions(df_eviction,df_median_housing_price,df_census,df_unemployment, df_future_data, months_ahead=3, plot_by_zip=False):
    '''Runs training models and forecast models, merges the results and predicts off of previous predictions made by the
    training models successively .'''

    #models

    top_down_forecast_df = run_top_down_forecast(df_eviction,df_median_housing_price, df_census, df_unemployment,df_future_data,months_ahead)
    random_forest_forecast_df = run_random_forest_forecast(df_eviction,df_median_housing_price, df_census, df_unemployment,df_future_data,months_ahead)

    #run training models. needs to be pickled.
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    random_forest_df, importance_dict = model_random_forest(eviction_median_housing,10,'auto')
    top_down_by_zip_df = top_down_estimation_by_zip(eviction_median_housing)


    merged_predictions = merge_all_models(top_down_forecast_df, random_forest_forecast_df)

    merged_training_data = merge_training_data (top_down_by_zip_df, random_forest_df)

    final_df = linear_regression_combination(merged_predictions, merged_training_data)

    if plot_by_zip:
        plot_by_zips(merged_training_data, final_df,zip_code=True)

    return final_df


def merge_all_models(top_down_forecast_df, random_forest_forecast_df):
    '''Merges model predictions together into a single dataframe of predictions, with each represented by a separate column'''
    merged_predictions_2 = pd.merge(top_down_forecast_df,random_forest_forecast_df,how='inner',\
                                    on= ['zip_code','month_year'], suffixes= ('','_random_forest'))

    return merged_predictions_2


def merge_training_data(top_down_by_zip_df, random_forest_df):
    '''Merges training predictions together into a single dataframe.'''
    merged_predictions_2 = pd.merge(top_down_by_zip_df,random_forest_df,how='inner',\
                                    left_on= ['Address_Zipcode','Month_Year','months_ahead'], right_on=['zip_code','month_year','months_ahead'],suffixes= ('','_random_forest'))
    merged_training_data = merged_predictions_2[merged_predictions_2.months_ahead==3]

    return merged_training_data


def linear_regression_combination(merged_predictions, merged_training_data):
    '''
    Fits linear model on the training data predictions, with predictions from each model
    representing a variable. The y is the actual eviction notice total for the specific zip/month

    The test data is similarly constructed, just with future predictions from the random_forest_forecast_fit_predict
    and top_down_forecast models.

    Output is a dataframe containing predicted number of evictions for each future month by SF ZIP.
    '''
    months = merged_predictions.month_year.tolist()
    zips = merged_predictions.zip_code.tolist()

    linear_predictions = []



    X_train= merged_training_data[['predicted_evictions_by_zip',\
                                            'predicted_evictions_random_forest']]
    y_train = merged_training_data['Eviction_Notice']

    X_test = merged_predictions[['zip_predicted','predicted_evictions']]



    #fitting model and making predictions, month by month
    lr = LinearRegression()
    model = lr.fit(X_train,y_train)
    predictions= model.predict(X_test)
    linear_predictions.extend(predictions)


    #building dataframe to store the prediction data from the linear regression above
    predictions_df = pd.DataFrame(data={'month_year':months,\
                'linear_predicted_evictions':linear_predictions,'zip_code':zips})

    predictions_df['month_year'] = pd.to_datetime(predictions_df['month_year'])

    #exporting to csv for ease of use
    merged_training_data.to_csv('past_evictions_by_zip_SF.csv')
    predictions_df.to_csv('future_evictions_by_zip_SF.csv')

    return predictions_df



if __name__ == '__main__':
    predicted_evictions  = predict_evictions(df_eviction,df_median_housing_price,df_census,df_unemployment, df_future_data, months_ahead=3, plot_by_zip=True)
