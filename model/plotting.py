import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyflux as pf
from fbprophet import Prophet

import logging
import datetime

from data_processing_refactor import transform_merge_data
from cross_validation import arima_by_zip_data_transform, arima_by_zip_months
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

from collections import defaultdict

# Modelling Algorithms

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')


def plot_by_zips(df, predictions_df, zip_code=False):
    '''If zip_code is true, plots eviction notices per each unique zip in the dataframe
    on a separate plot '''

    if zip_code:
        for zip_code_1 in df.zip_code.unique():
            zip_level_plot(df,predictions_df,zip_code=zip_code_1)


def zip_level_plot(original_df, predictions_df,zip_code):
    '''Transforming and merging the past data with the prediction data to make a single dataframe
    suitable for plotting. Use plt.show() to see all generated plots.'''

    predictions_filtered_df = predictions_df[predictions_df['zip_code']==zip_code]
    original_filtered_df = original_df[original_df['zip_code']==zip_code]
    predictions_filtered_df = predictions_filtered_df[['month_year','linear_predicted_evictions']]
    original_filtered_df = original_filtered_df[['Month_Year','Eviction_Notice','predicted_evictions_by_zip']]
    original_filtered_df = original_filtered_df.rename(columns={'Month_Year':'month_year','predicted_evictions_by_zip':'linear_predicted_evictions'})

    merged_df  = original_filtered_df.append(predictions_filtered_df)

    fig,ax = plt.subplots()
    ax.plot(merged_df['month_year'],merged_df['Eviction_Notice'],label='Actual Eviction Notices')
    ax.plot(merged_df['month_year'],merged_df['linear_predicted_evictions'],label='Predicted Eviction Notices')

    ax.set(title='Actual and Predicted Eviction Notices in SF ZIP code '+str(zip_code),\
            ylabel = 'Evictions', xlabel='Month of Eviction Notice')

    ax.legend()



if __name__=='__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    plot_models(eviction_median_housing,'plot_acf')
