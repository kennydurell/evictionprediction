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




def prophet_by_month(df):
    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)
    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()
    # params = [(2, 1, 2), 36095.90415324242]
    predictions_df= pd.DataFrame(np.random.randn(1, 3), columns=['month_year', 'actual_evictions', 'predicted_evictions'])

    i=0
    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
        (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
        (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

        y_train, y_test = transformed_df.iloc[train_indices].reset_index(inplace=False), transformed_df.iloc[test_indices].reset_index(inplace=False)
        for i,row in y_train.Eviction_Notice.iteritems():
            if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*4):
                y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()

        y_train = y_train.rename(columns={'Month_Year':'ds','Eviction_Notice':'y'})
        y_test = y_test.rename(columns={'Month_Year':'ds','Eviction_Notice':'y'})
        y_train = y_train[['ds','y']]
        y_test = y_test[['ds','y']]
        # print y_test
        # print y_train
        # break

        try:
            #predicted, actual = arimax_by_month_fit_predict(y_train,y_test,param)

            model = Prophet()
            model.fit(y_train)
            future = model.make_future_dataframe(periods=3,freq='M')
            predicted = model.predict(future)['yhat'].values.tolist()[-3:]
            actual = y_test.y.values.tolist()

            temp_df = pd.DataFrame({'actual_evictions':actual,\
                                        'predicted_evictions':predicted,'months_ahead':[1,2,3]})
            temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values
            predictions_df = predictions_df.append(temp_df, ignore_index=True)
            i+=1
            print i

        except Exception as e:
            logging.error(e, exc_info=True)

    # rmse =(mean_squared_error(predictions_df[1:].actual_evictions,predictions_df[1:].predicted_evictions))**.5
    predictions_df = predictions_df[1:]
    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])


    # three_ahead = predictions_df[predictions_df['months_ahead']==3]
    # predicted = three_ahead['predicted_evictions']
    # actual = three_ahead['actual_evictions']

    #rmse_arimax_by_month = (mean_squared_error(actual, predicted))**0.5

    return predictions_df

if __name__ == '__main__':
    processed_data = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    prophetic_df = prophet_by_month(processed_data)
