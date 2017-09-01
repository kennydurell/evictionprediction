import numpy as np
import pandas as pd
import datetime
from data_processing import data_processing_eviction, data_processing_housing, merge_data
from pandas.tools.plotting import autocorrelation_plot

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
df_median_housingprice_2 = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')
def baseline_model(df):
    #initializing lists and regressor
    y_true_values = []
    y_predicted_values = []

    zip_dict_true =defaultdict(list)
    zip_dict_predicted = defaultdict(list)
    rmse_final_dict = {}


    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    merged_sorted = df[['Month', 'Year', 'Month_S','Year_S','zip_code', \
                        'Month_Year','Eviction_Notice','CASANF0URN',\
                        'population']].sort_values(['Month_Year'])
    merged_sorted['Day_S'] = df['Month_Year'].dt.day
    merged_sorted = merged_sorted.dropna(subset=['population','CASANF0URN'])

    merged_sorted = merged_sorted.reset_index(drop=True)

    #creating X and y for regressor. Dropping unnecessary fields for splitting.
    y = merged_sorted.pop('Eviction_Notice')
    #X = merged_sorted.drop(['Month_Year','Month_S','Year_S','Day_S'], axis=1)

    #creating list of unique months in the data.
    months = merged_sorted[merged_sorted['Month_Year']>min(merged_sorted['Month_Year'])][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]

    #for loop allows for cross validation specific to time series, where values
    #that wouldn't be known at the moment of prediction (i.e. future values and even certain
    #summary values in the present) aren't used to predict/validate. result is an rmse for evictions by month
    #for all zips

    i=0
    for month in months_list:
        train_indices = merged_sorted[merged_sorted['Month_Year']<month].index
        test_indices = merged_sorted[merged_sorted['Month_Year']==month].index
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices].tolist()
        y_hat = [y_train.mean()for i in xrange(len(y_test))]
        y_predicted_values.extend(y_hat)
        y_true_values.extend(y_test)
        zip_counter=0
        for zip_code in merged_sorted.iloc[test_indices]['zip_code']:
            zip_dict_predicted[zip_code].append(y_hat[zip_counter])
            zip_dict_true[zip_code].append(y_test[zip_counter])
            zip_counter+=1
        i+=1
        print i

    for zip_code in zip_dict_true.keys():
        rmse_final_dict[zip_code] = (mean_squared_error(zip_dict_true[zip_code],zip_dict_predicted[zip_code]))**0.5

    return y_true_values, y_predicted_values, rmse_final_dict




if __name__ == '__main__':
    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_processed = data_processing_housing(df_median_housingprice_2)
    eviction_median_housing = merge_data(df_eviction_processed,df_median_housing_processed, df_census, df_unemployment)
    y_true_values, y_predicted_values, rmse_final_dict = baseline_model(eviction_median_housing)
    print rmse_final_dict
