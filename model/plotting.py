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



def plot_unprocessed_data (df,type_1):
    zip_list = df.Address_Zipcode.unique()

    if type_1 == 'plot_acf':
        for zip_code in zip_list:
            plot_acf_2(df,zip_code)

    else:
        for zip_code in zip_list:
            plot_time_series(df,zip_code)


def plot_processed_data (df,zip_code=False):

    if zip_code:
        for zip_code_1 in df.zip_code.unique():
            zip_level_plot(df,zip_code=zip_code_1)
    else:
        entire_city_plot(df)

def plot_time_series(df,zip_code):
    fig,ax = plt.subplots()
    plotting_df = df[df['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].set_index('Month_Year',inplace=False)
    ax.plot(plotting_df,label=zip_code)
    ax.legend()

def plot_acf_2(df,zip_code):
    plotting_df = df[df['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].set_index('Month_Year',inplace=False)
    plot_acf(plotting_df, title=zip_code)


def zip_level_plot(predictions_df,zip_code,columns_to_plot=['Month_Year','actual_evictions',\
                                        'linear_combination_predicted_evictions','zip_code']):

    zip_filtered_df = predictions_df[predictions_df[columns_to_plot[3]]==zip_code]

    fig,ax = plt.subplots()
    ax.plot(zip_filtered_df[columns_to_plot[0]],zip_filtered_df[columns_to_plot[1]],label='Actual Eviction Notices')
    ax.plot(zip_filtered_df[columns_to_plot[0]],zip_filtered_df[columns_to_plot[2]],label='Predicted Eviction Notices')

    ax.set(title='Actual and Predicted Eviction Notices in SF ZIP code '+ str(zip_code) + ' (ARIMAX)',\
            ylabel = 'Evictions', xlabel='Month of Eviction Notice')
    ax.legend()


def entire_city_plot(predictions_df,columns_to_plot=['month_year','actual_evictions',\
                                        'predicted_evictions','zip_code']):
    fig,ax = plt.subplots()
    ax.plot(predictions_df[columns_to_plot[0]],predictions_df[columns_to_plot[1]],\
                                                    label='Actual Eviction Notices')
    ax.plot(predictions_df[columns_to_plot[0]],predictions_df[columns_to_plot[2]],\
                                                    label='Predicted Eviction Notices')

    ax.set(title='Actual and Predicted Eviction Notices in San Francisco (ARIMAX)',\
                            ylabel = 'Evictions', xlabel='Month of Eviction Notice')
    ax.legend()


if __name__=='__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    plot_models(eviction_median_housing,'plot_acf')
