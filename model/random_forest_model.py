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

df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')



def model_random_forest(df, num_estimators, m_features, std=3):
    """Takes a dataframe of eviction data, runs it through a random forest model
        and output predictions 1, 2 and 3 months into the future by zip.

    Parameters:
    df -- processed eviction dataframe with all features added
    num_estimators - number of trees to use in the random forest model
    m_features - randomly selected maximum number of features each brannch is permitted to choose from
                    at each split
    std - standard deviations above the mean. Used in determining which datapoints
            in the training set are considered outliers.

    Output:
    predictions_df - predictions for evictions in each zip 1,2 and 3 months into the future
    importance_dict - dictionary of feature importances from the final random forest fit.
    """

    #initializing lists and regressor
    rfr = RandomForestRegressor(n_estimators = num_estimators, max_features=m_features )
    predictions_df= pd.DataFrame(np.random.randn(1, 5), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions','months_ahead'])
    importance_dict={}

    #clean/sort data for model
    merged_sorted = random_forest_model_data_cleaning(df)

    #creating X and y for regressor. Dropping unnecessary fields for splitting.
    y = merged_sorted.pop('Eviction_Notice')
    X = merged_sorted.drop(['Month_Year','Month_S','Year_S','Day_S'], axis=1)

    #creating list of unique months in the data.
    months_list = arima_by_zip_months(merged_sorted)
    months_ahead_list = [1,2,3]

    #for loop allows for cross validation specific to time series, where values
    #that wouldn't be known at the moment of prediction (i.e. future values and even certain
    #summary values in the present) aren't used to predict/validate. result is an rmse for evictions by month
    #for all zips
    i=0
    for months_ahead in months_ahead_list:
        for month in months_list:
            X_train, X_test, y_train, y_test,test_indices = \
                    random_forest_train_test(merged_sorted,X,y,month,months_ahead,std)

            rfr.fit(X_train,y_train)
            y_hat = rfr.predict(X_test).tolist()
            zips = merged_sorted.iloc[test_indices]['Address_Zipcode']
            temp_df = pd.DataFrame(data={'predicted_evictions':y_hat,'actual_evictions':y_test,\
                                    'zip_code': zips})

            temp_df['month_year']=month
            temp_df['months_ahead']=months_ahead
            predictions_df = predictions_df.append(temp_df,ignore_index=True)


            if month==months_list[-1]:
                importances = rfr.feature_importances_
                std_1 = np.std([tree.feature_importances_ for tree in rfr.estimators_],axis=0)
                indices_import = np.argsort(importances)[::-1]
                for f in range(X_train.shape[1]):
                    importance_dict[f+1] = \
                                [X_train.columns.tolist()[indices_import[f]],\
                                    importances[indices_import[f]],months_ahead]
            i+=1
            print i

    predictions_df=predictions_df[1:]
    predictions_df.month_year = pd.to_datetime(predictions_df.month_year)


    return predictions_df, importance_dict


def random_forest_model_data_cleaning(df):
    """Munges, selects relevant features, dummifies and cleans the eviction dataset for use in the random forest model."""

    #additional data processing to ensure it is in ascending datetime order, with most recent date at bottom
    merged_sorted = df[['Month','Year', 'Month_S','Year_S','Address_Zipcode','Month_Year','Eviction_Notice','CASANF0URN', 'CASANF0URN_unemployment_six_months_prior','capital_improvement_petition','capital_improvement_petition_six_months_prior','capital_improvement_petition_two_years_prior', 'Black_population_previous_year', 'median_sale_price_one_year_prior','median_sale_price_six_months_prior']].sort_values('Month_Year')

    merged_sorted['Day_S'] = df['Month_Year'].dt.day

    merged_sorted['CASANF0URN'] = \
                merged_sorted['CASANF0URN'].apply(lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['CASANF0URN_unemployment_six_months_prior'] = \
                merged_sorted['CASANF0URN_unemployment_six_months_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['capital_improvement_petition_two_years_prior'] = \
                merged_sorted['capital_improvement_petition_two_years_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['capital_improvement_petition'] =\
                merged_sorted['capital_improvement_petition'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['capital_improvement_petition_six_months_prior'] =\
                merged_sorted['capital_improvement_petition_six_months_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['capital_improvement_petition_two_years_prior'] =\
                merged_sorted['capital_improvement_petition_two_years_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['Black_population_previous_year'] =\
                merged_sorted['Black_population_previous_year'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['median_sale_price_one_year_prior'] =\
                merged_sorted['median_sale_price_one_year_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)
    merged_sorted['median_sale_price_six_months_prior'] =\
                merged_sorted['median_sale_price_six_months_prior'].apply\
                                            (lambda x:-1000 if pd.isnull(x) else x)

    #creating dummies for zip_codes
    zip_dummies = pd.get_dummies(merged_sorted['Address_Zipcode'])
    merged_sorted = pd.concat([merged_sorted,zip_dummies],axis=1)
    merged_sorted = merged_sorted.reset_index(drop=True)

    return merged_sorted


def random_forest_train_test(merged_sorted,X,y,month,months_ahead,std):
    """Sets up appropriate train/test split for the time series data"""

    train_indices = merged_sorted[merged_sorted['Month_Year']<\
                                (month-pd.offsets.MonthBegin(months_ahead-1))].index
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

    return X_train, X_test, y_train, y_test, test_indices


if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    random_forest_df_prior, importance_dict_prior = model_random_forest(eviction_median_housing,10,'auto')
    #rmse_final_dict = rf_cv(eviction_median_housing)
    #print rmse_final_dict
