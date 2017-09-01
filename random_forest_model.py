import numpy as np
import pandas as pd
import datetime
from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot
from baseline_model import baseline_model
from collections import defaultdict
import matplotlib.pyplot as plt

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier


from cross_validation import arima_by_zip_months

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

def model_random_forest(df, num_estimators, m_features, std=3):
    #initializing lists and regressor
    rfr = RandomForestRegressor(n_estimators = num_estimators, max_features=m_features )
    predictions_df= pd.DataFrame(np.random.randn(1, 4), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions'])


    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    merged_sorted = df[['Month', 'Year', 'Month_S','Year_S','Address_Zipcode', \
                        'Month_Year','Eviction_Notice','CASANF0URN', \
                        'CASANF0URN_unemployment_six_months_prior','Ground (landlord): Capital Improvement','Ground (landlord): Capital Improvement_six_months_prior','Ground (landlord): Capital Improvement_two_years_prior']].sort_values(['Month_Year'])
    merged_sorted['Day_S'] = df['Month_Year'].dt.day
    merged_sorted = merged_sorted.dropna(subset=['CASANF0URN',\
                        'CASANF0URN_unemployment_six_months_prior','Ground (landlord): Capital Improvement','Ground (landlord): Capital Improvement_six_months_prior','Ground (landlord): Capital Improvement_two_years_prior'])

    #creating dummies for zip_codes
    zip_dummies = pd.get_dummies(merged_sorted['Address_Zipcode'])
    merged_sorted = pd.concat([merged_sorted,zip_dummies],axis=1)
    merged_sorted = merged_sorted.reset_index(drop=True)

    #creating X and y for regressor. Dropping unnecessary fields for splitting.
    y = merged_sorted.pop('Eviction_Notice')
    X = merged_sorted.drop(['Month_Year','Month_S','Year_S','Day_S'], axis=1)

    #creating list of unique months in the data.
    months_list = arima_by_zip_months(merged_sorted)


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

        for zip_code in X_train['Address_Zipcode'].unique():
            indices_2 = X_train[X_train.Address_Zipcode==zip_code].index
            for i,value in y_train.iloc[indices_2].iteritems():
                if value>(y_train.iloc[indices_2].mean()+ y_train.iloc[indices_2].std()*std):
                    y_train.iloc[i]=y_train.iloc[indices_2].mean()

        X_train = X_train.drop('Address_Zipcode',axis=1)
        X_test = X_test.drop('Address_Zipcode',axis=1)

        rfr.fit(X_train,y_train)
        y_hat = rfr.predict(X_test).tolist()

        zips = merged_sorted.iloc[test_indices]['Address_Zipcode']
        temp_df = pd.DataFrame(data={'predicted_evictions':y_hat,'actual_evictions':y_test,\
                                'zip_code': zips})

        temp_df['month_year']=month
        predictions_df = predictions_df.append(temp_df,ignore_index=True)

        i+=1
        print i
        
    predictions_df=predictions_df[1:]
    predictions_df.month_year = pd.to_datetime(predictions_df.month_year)

    rmse_by_zip_dict = {}
    for zip_code in predictions_df.zip_code.unique():
        zip_filtered_df = predictions_df[predictions_df.zip_code==zip_code]
        rmse_by_zip_dict[zip_code] = (mean_squared_error(zip_filtered_df['actual_evictions'],zip_filtered_df['predicted_evictions']))**0.5


    return predictions_df, rmse_by_zip_dict


def rf_cv(df):
    predictions_df, rmse_final_dict = model_random_forest(df, num_estimators=10, m_features='auto',std=3)
    y_true, y_predict, baseline = baseline_model(df)
    comparison = rmse_final_dict
    second_dict ={}

    n_estimators = [10,40,80,200,1000]
    max_features =[2,4,'auto']
    std_list=[1,2,3,4,6,10]

    for estimator in n_estimators:
        for feature in max_features:
            for std in std_list:
                predictions_df, rmse_final_dict_cv = model_random_forest(df, num_estimators=estimator, m_features = feature, std=std)
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
    random_forest_df, rmse_by_zip_dict = model_random_forest(eviction_median_housing,1000,'auto')
    #rmse_final_dict = rf_cv(eviction_median_housing)
    #print rmse_final_dict
