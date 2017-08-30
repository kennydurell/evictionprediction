import numpy as np
import pandas as pd
import datetime
from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot
from baseline_model import baseline_model
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

def model_random_forest(df, num_estimators, m_features, std):
    #initializing lists and regressor
    y_true_values = []
    y_predicted_values = []
    rfr = RandomForestRegressor(n_estimators = num_estimators, max_features=m_features )
    zip_dict_true =defaultdict(list)
    zip_dict_predicted = defaultdict(list)
    rmse_final_dict = {}


    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    merged_sorted = df[['Month', 'Year', 'Month_S','Year_S','zip_code', \
                        'Month_Year','Eviction_Notice','CASANF0URN', \
                        'CASANF0URN_unemployment_six_months_prior','Estimate; RACE - One race - Black or African American_two_years_prior','Black_population_previous_year','percent_white_population_previous_year']].sort_values(['Month_Year'])
    merged_sorted['Day_S'] = df['Month_Year'].dt.day
    merged_sorted = merged_sorted.dropna(subset=['CASANF0URN',\
                        'CASANF0URN_unemployment_six_months_prior','percent_white_population_previous_year','Estimate; RACE - One race - Black or African American_two_years_prior','Black_population_previous_year'])

    merged_sorted = merged_sorted.reset_index(drop=True)

    #creating X and y for regressor. Dropping unnecessary fields for splitting.
    y = merged_sorted.pop('Eviction_Notice')
    X = merged_sorted.drop(['Month_Year','Month_S','Year_S','Day_S'], axis=1)

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
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices].tolist()
        for zip_code in X_train['zip_code'].unique():
            indices_2 = X_train[X_train.zip_code==zip_code].index
            for i,value in y_train.iloc[indices_2].iteritems():
                if value>(y_train.iloc[indices_2].mean()+ y_train.iloc[indices_2].std()*std):
                    y_train.iloc[i]=y_train.iloc[indices_2].mean()
        rfr.fit(X_train,y_train)
        y_hat = rfr.predict(X_test).tolist()
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


def rf_cv(df):
    y_true_values, y_predicted_values, rmse_final_dict = model_random_forest(df, num_estimators=10, m_features='auto',std=3)
    y_true, y_predict, baseline = baseline_model(df)
    comparison = rmse_final_dict
    second_dict ={}

    n_estimators = [10,40,80,200,1000]
    max_features =[2,4,'auto']
    std_list=[1,2,3,4,6,10]

    for estimator in n_estimators:
        for feature in max_features:
            for std in std_list:
                y_true_values, y_predicted_values, rmse_final_dict_cv = model_random_forest(df, num_estimators=estimator, m_features = feature, std=std)
                for zip_code in rmse_final_dict_cv.keys():
                    if rmse_final_dict_cv[zip_code] - comparison[zip_code] < 0:
                        baseline_diff = rmse_final_dict_cv[zip_code] - baseline[zip_code]
                        second_dict[zip_code]=(rmse_final_dict_cv[zip_code],baseline_diff,[estimator,feature,std])
                    else:
                        baseline_diff_2 = comparison[zip_code] - baseline[zip_code]
                        second_dict[zip_code] = (comparison[zip_code],baseline_diff_2,[10,'auto',3])
    return second_dict


if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    #y_true_values, y_predicted_values, rmse_final_dict = model_random_forest(eviction_median_housing,10,'auto')
    rmse_final_dict = rf_cv(eviction_median_housing)
    #print rmse_final_dict
