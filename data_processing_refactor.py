
# Handle table-like data and matrices
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#importing datasets
df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housingprice = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')


def eviction_datetime_shift (df_eviction):

    df_eviction['File Date'] = pd.to_datetime(df_eviction['File Date'])
    df_eviction['start_of_month_datetime'] = df_eviction['File Date']-pd.offsets.MonthBegin(1)
    df_eviction = df_eviction.groupby(['start_of_month_datetime', 'Eviction Notice Source Zipcode']).sum().reset_index()
    df_eviction['Year'] = df_eviction['start_of_month_datetime'].dt.year
    df_eviction['Month'] = df_eviction['start_of_month_datetime'].dt.month

    return df_eviction

def eviction_boolean_cleaning (df_eviction):

    df_eviction.replace(False,0,inplace=True)
    df_eviction.replace(True,1,inplace=True)
    df_eviction['Eviction_Notice'] = 1

    return df_eviction

def data_processing_eviction(df_eviction):
    '''Performs a variety of transformations on eviction data from DataSF:

    PARAMETERS
    ----------
    df_eviction: Pandas DataFrame of eviction data from here:'''

    boolean_cleaned = eviction_boolean_cleaning(df_eviction)
    datetime_shifted = eviction_datetime_shift(boolean_cleaned)

    return datetime_shifted

def housing_datetime_shift(df_median_housingprice):

    df_median_housingprice['period_begin']=pd.to_datetime(df_median_housingprice['period_begin'])
    df_median_housingprice['period_end']=pd.to_datetime(df_median_housingprice['period_end'])
    df_median_housingprice['one_month_prior']=df_median_housingprice['period_begin']-pd.offsets.MonthBegin(1)+pd.offsets.MonthEnd(1)
    df_median_housingprice['three_months_prior']=df_median_housingprice['period_begin']-pd.offsets.MonthBegin(3)+pd.offsets.MonthEnd(1)
    df_median_housingprice['six_months_prior']=df_median_housingprice['period_begin']-pd.offsets.MonthBegin(6)+pd.offsets.MonthEnd(1)
    df_median_housingprice['one_year_prior']=df_median_housingprice['period_begin']-pd.offsets.MonthBegin(12)+pd.offsets.MonthEnd(1)

    return df_median_housingprice

def housing_add_time_lags(df_median_housing_price,list_of_time_lags):

    for time_lag in list_of_time_lags:
        df_median_housing_price = pd.merge(df_median_housing_price,\
                                df_median_housing_price[['median_sale_price','zip_code','period_end']],\
                                how='left',left_on=['zip_code',time_lag],right_on=['zip_code','period_end'],\
                                suffixes=('','_'+time_lag))

    return df_median_housing_price

def housing_select_columns(df_median_housing_price,selected_columns):

    df_median_housing_price = df_median_housing_price[selected_columns]

    df_median_housing_price['zip_code']=df_median_housing_price['zip_code'].astype(str)

    return df_median_housing_price


def data_processing_housing(df_median_housing_price):
    '''Performs a variety of transformations on housing price data from Redfin.

    PARAMETERS
    ----------
    df_median_housing_price_2: Pandas data of median house prices in SF. From Redfin.
    '''

    datetime_shifted_df = housing_datetime_shift(df_median_housing_price)

    time_lagged_df = housing_add_time_lags(datetime_shifted_df, ['one_month_prior','three_months_prior',\
                                                                    'six_months_prior','one_year_prior'])
    columns_to_select = ['zip_code','period_begin',\
        'period_end','median_sale_price','one_month_prior',\
        'three_months_prior','six_months_prior','one_year_prior',\
        'median_sale_price_one_month_prior', 'median_sale_price_three_months_prior',\
        'median_sale_price_six_months_prior','median_sale_price_one_year_prior']

    selected_housing_df = housing_select_columns(time_lagged_df, columns_to_select)

    return selected_housing_df



def census_cleaning(df_census):

    df_census['population']=df_census['HC01_VC03'].astype(int)
    df_census['zip_code']= df_census['GEO.id2'].astype(str)
    df_census.drop(['GEO.id2','HC01_VC03'], axis=1,inplace=True)

    return df_census


def merged_data_datetime_shift(merged_df):

    merged_df = merged_df.drop(['median_sale_price'], axis=1)

    merged_df['Month_S'] = merged_df['Month'].map(int)
    merged_df['Year_S'] = merged_df['Year'].map(int)
    merged_df['Month_Year']= pd.to_datetime(dict(year=merged_df.Year_S, month=merged_df.Month_S, day=1))
    merged_df['one_month_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(1)
    merged_df['one_year_prior']=merged_df['Year_S']-1
    merged_df['two_years_prior']=merged_df['Year_S']-2
    merged_df['three_years_prior']=merged_df['Year_S']-3
    merged_df['four_years_prior']=merged_df['Year_S']-4
    merged_df['five_years_prior']=merged_df['Year_S']-5
    merged_df['datetime_six_months_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(6)
    merged_df['datetime_one_year_prior']=merged_df['Month_Year']-pd.Timedelta(days=365)
    merged_df = pd.merge(merged_df,merged_df[['Eviction_Notice','Month_Year','zip_code']],\
                                                how='left', left_on=['one_month_prior','zip_code'],\
                                                right_on= ['Month_Year','zip_code'], \
                                                suffixes=('','_one_month_prior'))

    return merged_df

def census_merge(merged_df, df_census):

    merged_df = pd.merge(merged_df,df_census[['zip_code','population','Year']],how='left',left_on=['zip_code', 'one_year_prior'], right_on=['zip_code','Year'], suffixes=('','_year_prior'))
    merged_df = pd.merge(merged_df,df_census[['zip_code','population','Year']],how='left',left_on=['zip_code', 'two_years_prior'], right_on=['zip_code','Year'], suffixes=('','_two_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['zip_code','population','Year']],how='left',left_on=['zip_code', 'three_years_prior'], right_on=['zip_code','Year'], suffixes=('','_three_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['zip_code','population','Year']],how='left',left_on=['zip_code', 'four_years_prior'], right_on=['zip_code','Year'], suffixes=('','_four_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['zip_code','population','Year']],how='left',left_on=['zip_code', 'five_years_prior'], right_on=['zip_code','Year'], suffixes=('','_five_years_prior'))
    return merged_df

def unemployment_merge(merged_df,df_unemployment):

    df_unemployment['DATE'] = pd.to_datetime(df_unemployment.DATE)
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_one_year_prior'], right_on=['DATE'], suffixes=('','_unemployment_year_prior'))
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_six_months_prior'], right_on=['DATE'], suffixes=('','_unemployment_six_months_prior'))
    return merged_df

def housing_merge(df_eviction_processed,df_median_housing_price_processed):
    eviction_housing_merged_df = pd.merge(df_eviction_processed,df_median_housing_price_processed,how='left',\
        left_on=['Eviction Notice Source Zipcode','start_of_month_datetime'],\
        right_on=['zip_code','period_begin'])
    return eviction_housing_merged_df

def transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment):
    '''Performs a variety of transformations and merges with eviction and housing price data

    PARAMETERS
    ----------
    df_eviction: Pandas DataFrame of eviction data from here: https://data.sfgov.org/Housing-and-Buildings/Eviction-Notices/5cei-gny5
    df_median_housing_price: Pandas data of median house prices in SF. From Redfin.
    df_census: aggregated data from the American Community Survey: https://www.census.gov/programs-surveys/acs/
    df_unemployment: unemployment percentage by month for the city of SF: https://fred.stlouisfed.org/series/CASANF0URN'''

    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_price_processed = data_processing_housing(df_median_housingprice)
    eviction_housing_merged_df = housing_merge(df_eviction_processed,df_median_housing_price_processed)
    datetime_shifted_df = merged_data_datetime_shift(eviction_housing_merged_df)
    cleaned_census_df = census_cleaning(df_census)
    census_merged_df = census_merge(datetime_shifted_df,cleaned_census_df)
    unemployment_merged_df= unemployment_merge(census_merged_df,df_unemployment)
    final_merged_df = unemployment_merged_df

    return final_merged_df


    # merged_df = pd.merge(merged_df[['Month', 'Year', 'zip_code', 'median_sale_price_one_month_prior',\
    #     u'median_sale_price_three_months_prior',u'median_sale_price_six_months_prior',\
    #     u'median_sale_price_one_year_prior','Eviction_Notice', 'Month_Year', 'Month_S', 'Year_S','one_month_prior']],\
    #     merged_df[['Eviction_Notice','Month_Year','zip_code']], how='left', \
    #     left_on=['one_month_prior','zip_code'],right_on= ['Month_Year','zip_code'], suffixes=('','_one_month_prior'))
    # merged_df = merged_df.dropna(subset = ['period_begin','median_sale_price_one_month_prior',\
    #     'median_sale_price_one_year_prior', 'median_sale_price_six_months_prior','median_sale_price_three_months_prior']).sort_values('period_begin')

    #




def time_window_aggregate(df, feature,func,time_window):
    '''
    Calculates the aggregate value of a column for a given time_window starting
    before each point
    PARAMETERS
    ----------
    df: Pandas DataFrame
    feature: column name of feature to shift
    time_window: time window to aggregate over, use convention used in Pandas
                 reindex or rolling:http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    func: Aggregate function to use
    '''
    temp = pd.DataFrame(df[feature].shift(1))
    if func == 'sum':
        result = temp.rolling(time_window,min_periods=3).sum()
    elif func == 'std':
        result = temp.rolling(time_window,min_periods=3).std()
    elif func == 'mean':
        result = temp.rolling(time_window,min_periods=3).mean()
    elif func == 'min':
        result = temp.rolling(time_window,min_periods=3).min()
    elif func == 'max':
        result = temp.rolling(time_window,min_periods=3).max()
    else:
        print 'Aggregate function {} does not exist'.format(func)
        return None
    col_name = '{}_{}_{}'.format(feature,time_window,func)
    df[col_name] = result
    return df


if __name__ =='__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
