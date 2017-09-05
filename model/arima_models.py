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


def arimax_by_zip (df,zip_param_dictionary):
    """Take a dataframe of eviction data, run it through an arima model
        and output predictions 1, 2 and 3 months into the future by zip. This is used
        for cross-validaton/model selection, so it assumes that this is data from the past
        and there is an actual eviction notice number to compare the final prediction value to.

    Parameters:
    df -- processed eviction dataframe with all features added
    zip_param_dictionary - dictionary of ARIMA parameters to use for each ZIP

    Output:
    predictions_df - predictions for each zip 1,2 and 3 months into the future
    rmse_dict - root mean square error comparing predictions to actuals for past data
    """

    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)

    predictions_df= pd.DataFrame(np.random.randn(1, 5), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions','months_ahead'])

    i_2=0
    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

        for zip_code,value in zip_param_dictionary.iteritems():
            y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
            if zip_code in y_train.Address_Zipcode.values and zip_code in y_test.Address_Zipcode.values:
                y_train = y_train[y_train['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN']].reset_index(inplace=False)
                y_test = y_test[y_test['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN']].reset_index(inplace=False)

                for i,row in y_train.Eviction_Notice.iteritems():
                    if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*value[2]):
                        y_train.Eviction_Notice.iloc[i]=(y_train.Eviction_Notice.mean())


                y_train = y_train.set_index(['Month_Year'], inplace=False)
                y_test = y_test.set_index(['Month_Year'], inplace=False)

                try:
                    predicted, actual = arimax_by_zip_fit_predict(y_train,y_test,value[0])
                    temp_df = pd.DataFrame(data={'predicted_evictions':predicted,'actual_evictions':actual,\
                                            'zip_code': zip_code,'months_ahead':[1,2,3]})
                    temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values
                    predictions_df = predictions_df.append(temp_df,ignore_index=True)
                    i_2+=1
                    print i_2

                except Exception as e:
                    logging.error(e, exc_info=True)

    predictions_df = predictions_df[1:]
    predictions_df.month_year = pd.to_datetime(predictions_df.month_year)
    rmse_dict = rmse_calculation(predictions_df)

    print 'ARIMAX by ZIP'
    return predictions_df, rmse_dict


def rmse_calculation(predictions_df):
    """Performs RMSE calculaton for a dataframe containing predictions and actual eviction data."""

    rmse_dict = {}

    for zip_code in predictions_df.zip_code.unique():
        zip_filtered_df = predictions_df[predictions_df.zip_code==zip_code]
        rmse_dict[zip_code] = (mean_squared_error(zip_filtered_df['actual_evictions'],zip_filtered_df['predicted_evictions']))**0.5
    return rmse_dict


def arimax_by_zip_fit_predict(y_train,y_test,param):
    """Fits and predicts 1,2 and 3 months into the future for the given eviction dataset.
     Exogenous variable is used to perform a linear regression underlying the ARIMA fit. """

    model = pf.ARIMAX(data=y_train, formula='Eviction_Notice~1+CASANF0URN',
          ar=param[0], ma=param[2])
    model.fit()
    y_hat = model.predict(h=3,oos_data=y_test)
    predicted = y_hat.Eviction_Notice.values.tolist()
    actual = y_test.Eviction_Notice.values.tolist()

    return predicted, actual


def top_down_estimation_by_zip(original_df):
    """Takes processed eviction dataframe and first runs it through an arima model, predicting total eviction notices
    by month.
    Then, this function proceeds to decompose the aggregate predictions into zip code component parts,
    each based on an average of previous proportions of the total evictions. See here if you're looking
    for more information about this approach: https://www.otexts.org/fpp/9/4

    This is a cross-validation/model selection function.
    Output is a dataframe with predicted and actual values, by zip, side by side.

    Parameters:
    original_df -- processed eviction dataframe with all features added

    Output:
    top_down_by_zip_df - predictions for each zip 1,2 and 3 months into the future, at successive points in time.
    """


    predictions_by_month_df,rmse = arimax_by_month(original_df,(2,1,2),2)

    zip_perc_df= pd.DataFrame(np.random.randn(1, 4), columns=['month_for_zips','Address_Zipcode','perc_of_month','months_ahead_2'])
    months_ahead_list = [1,2,3]

    for months_ahead in months_ahead_list:
        percentage_of_month_df = pd.merge(original_df[['Month_Year','Address_Zipcode',\
        'Eviction_Notice']],predictions_by_month_df[predictions_by_month_df['months_ahead']==months_ahead],\
        how = 'left',left_on='Month_Year',right_on='month_year',suffixes=('','_for_month'))
        percentage_of_month_df.dropna(inplace=True)
        percentage_of_month_df['perc_of_month']=percentage_of_month_df['Eviction_Notice']/percentage_of_month_df['actual_evictions']
        months_list = pd.to_datetime(percentage_of_month_df[percentage_of_month_df.Month_Year>\
            (min(percentage_of_month_df.Month_Year)+pd.offsets.MonthBegin(3))]['month_year'].unique())

        i=0

        for month in months_list:
            indices = pd.notnull(percentage_of_month_df[percentage_of_month_df.Month_Year<(month-pd.offsets.MonthBegin(months_ahead-1))]['perc_of_month'])
            non_null_percentages_df = percentage_of_month_df[percentage_of_month_df.Month_Year<(month-pd.offsets.MonthBegin(months_ahead-1))].loc[indices]

            group_by_zip_df = non_null_percentages_df.groupby(['Address_Zipcode','months_ahead']).mean().reset_index()[['Address_Zipcode','perc_of_month']]
            group_by_zip_df['month_for_zips'] = month
            group_by_zip_df['months_ahead_2'] = months_ahead

            zip_perc_df = zip_perc_df.append(group_by_zip_df, ignore_index=True)


            i+=1
            print i

    zip_perc_df = zip_perc_df[1:]

    percentage_of_month_2_df = pd.merge(original_df[['Month_Year','Address_Zipcode',\
    'Eviction_Notice']],predictions_by_month_df,\
        how = 'left',left_on='Month_Year',right_on='month_year',suffixes=('','_for_month'))

    zip_perc_df.month_for_zips = pd.to_datetime(zip_perc_df.month_for_zips)
    final_df = pd.merge(percentage_of_month_2_df,zip_perc_df,left_on=['Month_Year','Address_Zipcode','months_ahead'],right_on=['month_for_zips','Address_Zipcode','months_ahead_2'],suffixes=('','_averaged_over_past_months'))
    final_df['predicted_evictions_by_zip'] = final_df['perc_of_month']*final_df['predicted_evictions']


    top_down_by_zip_df = final_df
    print 'top down zip'
    return top_down_by_zip_df


def arimax_by_month_fit_predict(y_train,y_test,month,param):
    """The fit_predict function for the ARIMAX by month eviction model.
    Uses exogenous variable CASANF0URN (unemployment rate in previous year) along with
    the standard time lagged AR and MA features.

    Parameters:
    y_train
    y_test
    month -- month being predicted, along with the two successive months
    param -- the AR, I and MA parameters for the ARIMA model.

    Output:
    temp_df - ARIMAX predictions of eviction totals for three successive, future months
    """

    model = pf.ARIMAX(data=y_train,formula='Eviction_Notice~1+CASANF0URN',ar=param[0], ma=param[2])
    model.fit()
    y_hat = model.predict(h=3,oos_data=y_test)

    actual = y_test.Eviction_Notice.values.tolist()
    predicted = y_hat.Eviction_Notice.values.tolist()

    temp_df = pd.DataFrame({'actual_evictions':actual,\
                                'predicted_evictions':predicted,'months_ahead':[1,2,3]})
    temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values


    return temp_df


def arimax_by_month_train_test(transformed_df,month,std):
    """
    Train test split for ARIMAX model. Does both time series train/test split as well
    as mean imputing for outliers in the training set.

    Parameters:
    transformed_df -- processed eviction dataframe with all features added
    month -- month in the future to predict. Prevents data leakage by ensuring data from the future is not used to improve predictions about past data.
    std -- adjusts for outliers in training data. Number of standard deviations from the mean a value must be to be considered an outlier
    and thus imputed with a mean.

    Output:
    y_train, y_test - train and test dataframes split into past (train) and future (test).
    """


    train_indices = transformed_df[transformed_df['Month_Year']<month].index
    test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
    (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
    (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

    y_train, y_test = transformed_df.iloc[train_indices].reset_index(inplace=False), transformed_df.iloc[test_indices].reset_index(inplace=False)

    for i,row in y_train.Eviction_Notice.iteritems():
        if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*std):
            y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()

    return y_train, y_test



def arimax_by_month (df,param,std):
    """
    ARIMAX model for aggregate eviction notice monthly prediction.
    Used in conjunction with top_down_estimation_by_zip function
    to predict at the ZIP level via decomposition of the total predicted value.

    Parameters:
    df -- processed eviction dataframe with all features added
    param - parameters to use in the arimax model.
    std -- adjusts for outliers in training data. Number of standard deviations from the mean a value must be to be considered an outlier
    and thus imputed with a mean.

    Output:
    predictions_by_month_df - dataframe with predictions by month.
    Also includes a column indicated how far ahead each prediction was made,
    based on the date range of the training set during the fit.

    rmse_arimax_by_month - rmse of all the datapoints predicted three months into the future.
    """

    predictions_df= pd.DataFrame(np.random.randn(1, 3),\
            columns=['month_year', 'actual_evictions', 'predicted_evictions'])

    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)
    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()

    i=0
    for month in months_list:
        y_train, y_test = arimax_by_month_train_test(transformed_df,month,std)

        y_train = y_train[['Month_Year','Eviction_Notice','CASANF0URN']].set_index(['Month_Year'],inplace=False)
        y_test = y_test[['Month_Year','Eviction_Notice','CASANF0URN']].set_index(['Month_Year'],inplace=False)

        try:
            temp_predictions_df = arimax_by_month_fit_predict(y_train,y_test,month,param)
            predictions_df = predictions_df.append(temp_predictions_df, ignore_index=True)

        except Exception as e:
            logging.error(e, exc_info=True)

    predictions_df = predictions_df[1:]
    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])

    three_ahead = predictions_df[predictions_df['months_ahead']==3]
    predicted = three_ahead['predicted_evictions']
    actual = three_ahead['actual_evictions']

    rmse_arimax_by_month = (mean_squared_error(actual, predicted))**0.5

    return predictions_df, rmse_arimax_by_month


if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    top_down_by_zip_df = top_down_estimation_by_zip(eviction_median_housing)
    zip_dict = {'94102': [(2, 1, 0),(1,0,0,6),4],
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
