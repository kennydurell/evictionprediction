
# Handle table-like data and matrices
import numpy as np
import pandas as pd
import datetime

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

# Visualization
from fbprophet import Prophet
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

!cd '/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction'
df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housingprice_2 = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_buyout = pd.read_csv('Buyout_agreements.csv')
df_request = pd.read_csv('Requests_for_Information_Regarding_Protected_Status_Related_to_Owner_Move-In_Evictions.csv')


def data_processing_eviction(df_eviction):
    
    df_eviction_copy = df_eviction.copy()
    df_eviction_copy.replace(False,0,inplace=True)
    df_eviction_copy.replace(True,1,inplace=True)
    df_eviction_copy['Eviction_Notice'] = 1
    df_eviction_copy['File Date'] = pd.to_datetime(df_eviction_copy['File Date'])
    df_eviction_copy['Year'] = df_eviction_copy['File Date'].dt.year
    df_eviction_copy['Month'] = df_eviction_copy['File Date'].dt.month
    df_eviction_copy['Day'] = df_eviction_copy['File Date'].dt.day
    df_eviction_copy['start_of_month_datetime'] = df_eviction_copy['File Date']-pd.offsets.MonthBegin(1)
    return df_eviction_copy

def data_processing_housing(df_median_housingprice_2):

    df_median_housingprice_2['period_begin']=pd.to_datetime(df_median_housingprice_2['period_begin'])
    df_median_housingprice_2['period_end']=pd.to_datetime(df_median_housingprice_2['period_end'])
    df_median_housingprice_2['one_month_prior']=df_median_housingprice_2['period_begin']-pd.offsets.MonthBegin(1)+pd.offsets.MonthEnd(1)
    df_median_housingprice_2['three_months_prior']=df_median_housingprice_2['period_begin']-pd.offsets.MonthBegin(3)+pd.offsets.MonthEnd(1)
    df_median_housingprice_2['six_months_prior']=df_median_housingprice_2['period_begin']-pd.offsets.MonthBegin(6)+pd.offsets.MonthEnd(1)
    df_median_housingprice_2['one_year_prior']=df_median_housingprice_2['period_begin']-pd.offsets.MonthBegin(12)+pd.offsets.MonthEnd(1)
    df_median_housingprice_2 = pd.merge(df_median_housingprice_2,df_median_housingprice_2[['median_sale_price','zip_code','period_end']]\
          ,how='left',left_on=['zip_code','one_month_prior'],right_on=['zip_code','period_end'],\
         suffixes=('','_one_month_prior'))

    df_median_housingprice_2 = pd.merge(df_median_housingprice_2,\
        df_median_housingprice_2[['median_sale_price','zip_code','period_end']],\
        how='left', left_on=['zip_code','three_months_prior'],right_on=['zip_code','period_end'],\
        suffixes=('','_three_months_prior'))

    df_median_housingprice_2 = pd.merge(df_median_housingprice_2,\
        df_median_housingprice_2[['median_sale_price','zip_code','period_end']],\
        how='left', left_on=['zip_code','six_months_prior'],right_on=['zip_code','period_end'],\
        suffixes=('','_six_months_prior'))

    df_median_housingprice_2 = pd.merge(df_median_housingprice_2,\
        df_median_housingprice_2[['median_sale_price','zip_code','period_end']],\
        how='left', left_on=['zip_code','one_year_prior'],right_on=['zip_code','period_end'],\
        suffixes=('','_one_year_prior'))

    df_median_housingprice_2['zip_code']=df_median_housingprice_2['zip_code'].astype(str)

    df_median_housingprice_2 = df_median_housingprice_2[['zip_code','period_begin',\
        'period_end','median_sale_price','one_month_prior',\
        'one_month_prior','three_months_prior','six_months_prior','one_year_prior',\
        'median_sale_price_one_month_prior', 'median_sale_price_three_months_prior',\
        'median_sale_price_six_months_prior','median_sale_price_one_year_prior']]

    return df_median_housingprice_2

def merge_data(df_eviction,df_median_housing_price_2):

    merged_df = pd.merge(df_eviction,df_median_housing_price_2,how='left',\
        left_on=['Eviction Notice Source Zipcode','start_of_month_datetime'],\
        right_on=['zip_code','period_begin'])

    merged_df = merged_df.dropna(subset = ['period_begin','median_sale_price_one_month_prior',\
        'median_sale_price_one_year_prior', 'median_sale_price_six_months_prior','median_sale_price_three_months_prior']).sort_values('period_begin')

    merged_df = merged_df.drop(['Constraints Date','Supervisor District ', \
        'Neighborhoods - Analysis Boundaries','Location', 'Supervisor District',\
        'Day','median_sale_price' ], axis=1)

    merged_df = merged_df.groupby(['Month','Year','zip_code','median_sale_price_one_month_prior',\
        'median_sale_price_three_months_prior','median_sale_price_six_months_prior',\
        'median_sale_price_one_year_prior']).sum().reset_index()

    return merged_df


if __name__ =='__main__':
    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_processed = data_processing_housing(df_median_housingprice_2)
    eviction_median_housing = merge_data(df_eviction_processed,df_median_housing_processed)
    eviction_median_housing.info()
