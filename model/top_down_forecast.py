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


def run_top_down_forecast(df_eviction,df_median_housing_price, df_census, df_unemployment,df_future_data, months_ahead=3):
    '''run each step of the forecasting model in order. Parameters are all unprocessed dataframes.'''

    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    future_df = future_data_processing(df_future_data)
    predictions_by_month,y_hat = arimax_by_month_forecast(eviction_median_housing,months_ahead,future_df)
    top_down_prediction_df = top_down_forecast(eviction_median_housing,predictions_by_month,months_ahead)
    return top_down_prediction_df


def top_down_forecast_data_processing(original_df,predictions_by_month):
    """
    Takes a dataframe of processed eviction data and predictions from an ARIMAX
    model at the month level. Returns a dataframe grouped by ZIP with a column
    containing the average percentage of total evictions each ZIP represented

    Parameters:
    original_df -- processed eviction dataframe with all features added
    predictions_by_month - dataframe with total evictions predicted by month for
    the number of months specified by the user.

    Output:
    group_by_zip_df -- dataframe with zip_code and perc_of_month (average percentage of monthly
    evictions) columns

    """
    transformed_df=original_df.groupby('Month_Year').sum().reset_index()\
                                            [['Month_Year','Eviction_Notice']]

    percentage_of_month_df = pd.merge(original_df[['Month_Year','Address_Zipcode',\
                        'Eviction_Notice']],transformed_df,how = 'left',left_on='Month_Year',\
                                    right_on='Month_Year',suffixes=('','_for_month'))

    percentage_of_month_df.dropna(inplace=True)

    percentage_of_month_df['perc_of_month']=\
        percentage_of_month_df['Eviction_Notice']/percentage_of_month_df['Eviction_Notice_for_month']

    group_by_zip_df = percentage_of_month_df.groupby('Address_Zipcode').mean().reset_index()[['Address_Zipcode','perc_of_month']]

    return group_by_zip_df


def top_down_forecast(original_df,predictions_by_month,months_ahead):
    """
    Takes a dataframe of processed eviction data and predictions from an ARIMAX
    model at the month level, as well as the number of months into the future to
    predict.

    Parameters:
    df -- processed eviction dataframe with all features added
    predictions_by_month - dataframe with total evictions predicted by month for
    the number of months specified by the user.
    months_ahead - user-specified number of months to predict out to

    Output:
    zip_perc_df (zip percentage df) - dataframe with percent of month column as
    well as predictions for each ZIP for each month and the total number of evictions
    predicted for that month.
    """

    group_by_zip_df = top_down_forecast_data_processing(original_df,predictions_by_month)

    zip_perc_df= pd.DataFrame(np.random.randn(1, 4), columns=['zip_predicted','zip_code','perc_of_month','month_year'])

    for month in range(months_ahead):
        array_of_predictions = np.asarray([predictions_by_month.iloc[month]['predicted_evictions']]*group_by_zip_df.shape[0])
        zip_predicted = np.asarray(group_by_zip_df.perc_of_month*array_of_predictions)
        temp_df = pd.DataFrame({'total_evictions':array_of_predictions,'zip_predicted':zip_predicted,'zip_code':np.asarray(group_by_zip_df.Address_Zipcode), 'perc_of_month':np.asarray(group_by_zip_df.perc_of_month)})
        temp_df['month_year']=predictions_by_month.iloc[month]['month_year']
        zip_perc_df = zip_perc_df.append(temp_df,ignore_index=True)

    zip_perc_df = zip_perc_df[1:]
    zip_perc_df['month_year']=pd.to_datetime(zip_perc_df['month_year'])

    return zip_perc_df



def arimax_by_month_train_test(model_data,future_data,months_ahead):
    """
    Takes a dataframe of processed eviction data as well as a set amount of lagged
    variables about future datapoints and how many months ahead to include in the prediction.

    Parameters:
    model_data -- processed eviction dataframe with all features added
    future_data - dataframe with endogenous data for set time period
    months_ahead - user-specified number of months to predict out to

    Output:
    y_train
    y_test
    months_list - list of all months included in the prediction.
    """
    transformed_df = arima_by_zip_data_transform(model_data)
    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()

    y_train = transformed_df


    y_test = future_data[['Month_Year','Eviction_Notice','CASANF0URN']]

    now = datetime.datetime.now()
    this_month = pd.to_datetime(str(now.year)+ '-'+ str(now.month) + '-'+str(1))
    y_test=y_test[(y_test.Month_Year>this_month)&(y_test.Month_Year<(this_month+pd.offsets.MonthBegin(months_ahead+1)))].sort_values('Month_Year')
    months_list = np.asarray(y_test.Month_Year)

    y_test = y_test.set_index(['Month_Year'],inplace=False)


    for i,row in y_train.Eviction_Notice.iteritems():
        if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*3):
            y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()

    y_train = y_train[['Month_Year','Eviction_Notice','CASANF0URN']].set_index(['Month_Year'],inplace=False)

    return y_train, y_test,months_list


def arimax_by_month_forecast_fit_predict (y_train,y_test,months_ahead, months_list):
    """
    Takes in training and test data formatted as time series with an exogenous variable, CASANF0URN, which
    which represents unemployment rate in San Francisco from the previous year.

    Parameters:
    y_train
    y_test
    months_ahead - user specified months ahead to predict
    months_list -

    Output:
    zip_perc_df (zip percentage df) - dataframe with percent of month column as
    well as predictions for each ZIP for each month and the total number of evictions
    predicted for that month.
    """
    predictions_df= pd.DataFrame(np.random.randn(1, 2),\
        columns=['month_year', 'predicted_evictions'])

    now = datetime.datetime.now()

    model = pf.ARIMAX(data=y_train,formula='Eviction_Notice~1+CASANF0URN',ar=2, ma=2)
    model.fit()
    y_hat = model.predict(h=months_ahead,oos_data=y_test)

    predicted = y_hat.Eviction_Notice.values.tolist()
    temp_df = pd.DataFrame({'predicted_evictions':predicted,'months_ahead':range(1,months_ahead+1)})
    temp_df['month_year']=months_list

    predictions_df = predictions_df.append(temp_df, ignore_index=True)
    predictions_df = predictions_df[1:]
    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])

    return predictions_df, y_hat


def arimax_by_month_forecast(model_data,months_ahead,future_data):
    '''Runs the train_test and fit_predict functions for the arimax by month prediction model.
    Parameters
    - processed eviction data(model_data)
    - months forward to predict (months_ahead)
    - exogenous variables lagged from future dates corresponding to the months_ahead (future_data)'''

    y_train, y_test,months_list = arimax_by_month_train_test(model_data,future_data,months_ahead)
    predictions_df, y_hat = arimax_by_month_forecast_fit_predict (y_train,y_test,months_ahead, months_list)

    return predictions_df, y_hat

def future_data_processing(df):
    '''Processing the exogenous features for the future prediction months to align with the model's fomrat'''

    future_df = transform_merge_data(df,df_median_housing_price, df_census, df_unemployment)
    future_transform_df = arima_by_zip_data_transform(future_df)
    future_transform_df=future_transform_df.groupby('Month_Year').sum().reset_index()

    return future_transform_df

if __name__=='__main__':

    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    future_df = future_data_processing(df_future_data)
    predictions_by_month,y_hat = arimax_by_month_forecast(eviction_median_housing,3,future_df)
    top_down_forecast_df = top_down_forecast(eviction_median_housing,predictions_by_month,3)
