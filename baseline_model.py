import numpy as np
import pandas as pd
import datetime
from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot
import logging

from collections import defaultdict

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# df_eviction = pd.read_csv('/home/ubuntu/eviction_data/Eviction_Notices.csv')
# df_median_housingprice_2 = pd.read_csv('/home/ubuntu/eviction_data/med_sp_zip_code_sf_ca (1).csv')

df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')

def baseline_model(df):
    #initializing lists and regressor
    y_true_values = defaultdict(list)
    y_predicted_values = defaultdict(list)
    rmse_final_dict = {}

    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    transformed_df = df[['Month', 'Year', 'Month_S','Year_S','Address_Zipcode', \
                        'Month_Year','Eviction_Notice','CASANF0URN',\
                        'population']].sort_values(['Month_Year'])
    transformed_df['Day_S'] = df['Month_Year'].dt.day
    # transformed_df = transformed_df.dropna(subset=['population','CASANF0URN'])

    transformed_df = transformed_df.reset_index(drop=True)


    #creating list of unique months in the data.
    months = transformed_df[transformed_df['Month_Year']>(min(transformed_df['Month_Year'])+pd.offsets.MonthBegin(3))][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]

    #for loop allows for cross validation specific to time series, where values
    #that wouldn't be known at the moment of prediction (i.e. future values and even certain
    #summary values in the present) aren't used to predict/validate. result is an rmse for evictions by month
    #for all zips

    i=0
    predictions_df= pd.DataFrame(np.random.randn(1, 5), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions','months_ahead'])

    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

        y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]

        zip_list = y_train.Address_Zipcode.unique()

        for zip_code in zip_list:
            y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
            if zip_code in y_train.Address_Zipcode.values and zip_code in y_test.Address_Zipcode.values:
                y_train = y_train[y_train['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].reset_index(inplace=False)
                y_test = y_test[y_test['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].reset_index(inplace=False)
                try:
                    predicted = [y_train.Eviction_Notice.mean()for i in xrange(len(y_test))]
                    actual = y_test.Eviction_Notice.tolist()
                    temp_df = pd.DataFrame(data={'predicted_evictions':predicted,'actual_evictions':actual,\
                                            'zip_code': zip_code,'months_ahead':[1,2,3]})
                    temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values
                    predictions_df = predictions_df.append(temp_df,ignore_index=True)

                    i+=1
                    print i

                except Exception as e:
                    logging.error(e, exc_info=True)


    months_ahead_list=[1,2,3]
    for months_ahead in months_ahead_list:
        rmse_temp_dict = {}
        months_ahead_filtered = predictions_df[predictions_df['months_ahead']==months_ahead]
        zip_list = months_ahead_filtered.zip_code.unique()
        for zip_code in zip_list:
            zip_filtered = months_ahead_filtered[months_ahead_filtered.zip_code==zip_code]
            rmse_temp_dict[zip_code] = (mean_squared_error(zip_filtered['actual_evictions'],zip_filtered['predicted_evictions']))**0.5

        rmse_final_dict[months_ahead]=rmse_temp_dict

    return predictions_df[1:], rmse_final_dict


def baseline_rolling_model(df):
    #initializing lists and regressor
    y_true_values = defaultdict(list)
    y_predicted_values = defaultdict(list)
    rmse_final_dict = {}

    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    transformed_df = df[['Month', 'Year', 'Month_S','Year_S','Address_Zipcode', \
                        'Month_Year','Eviction_Notice','CASANF0URN',\
                        'population']].sort_values(['Month_Year'])
    transformed_df['Day_S'] = df['Month_Year'].dt.day
    # transformed_df = transformed_df.dropna(subset=['population','CASANF0URN'])

    transformed_df = transformed_df.reset_index(drop=True)


    #creating list of unique months in the data.
    months = transformed_df[transformed_df['Month_Year']>(min(transformed_df['Month_Year'])+pd.offsets.MonthBegin(12))][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]

    #for loop allows for cross validation specific to time series, where values
    #that wouldn't be known at the moment of prediction (i.e. future values and even certain
    #summary values in the present) aren't used to predict/validate. result is an rmse for evictions by month
    #for all zips

    i=0
    predictions_df= pd.DataFrame(np.random.randn(1, 5), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions','months_ahead'])

    for month in months_list:
        train_indices = transformed_df[(transformed_df['Month_Year']<month)&(transformed_df['Month_Year']>(month-pd.offsets.MonthBegin(4)))].index
        test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
            (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

        y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]

        zip_list = y_train.Address_Zipcode.unique()

        for zip_code in zip_list:
            y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
            if zip_code in y_train.Address_Zipcode.values and zip_code in y_test.Address_Zipcode.values:
                y_train = y_train[y_train['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].reset_index(inplace=False)
                y_test = y_test[y_test['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice']].reset_index(inplace=False)
                try:
                    predicted = [y_train.Eviction_Notice.mean() for i in xrange(len(y_test))]
                    actual = y_test.Eviction_Notice.tolist()
                    temp_df = pd.DataFrame(data={'predicted_evictions':predicted,'actual_evictions':actual,\
                                            'zip_code': zip_code,'months_ahead':[1,2,3]})
                    temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values
                    predictions_df = predictions_df.append(temp_df,ignore_index=True)

                    i+=1
                    print i

                except Exception as e:
                    logging.error(e, exc_info=True)


    months_ahead_list=[1,2,3]
    for months_ahead in months_ahead_list:
        rmse_temp_dict = {}
        months_ahead_filtered = predictions_df[predictions_df['months_ahead']==months_ahead]
        zip_list = months_ahead_filtered.zip_code.unique()
        for zip_code in zip_list:
            zip_filtered = months_ahead_filtered[months_ahead_filtered.zip_code==zip_code]
            rmse_temp_dict[zip_code] = (mean_squared_error(zip_filtered['actual_evictions'],zip_filtered['predicted_evictions']))**0.5

        rmse_final_dict[months_ahead]=rmse_temp_dict

    return predictions_df[1:], rmse_final_dict


if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    predictions_df, rmse_final_dict = baseline_rolling_model(eviction_median_housing)
