
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

# Visualization



# df_eviction = pd.read_csv('/home/ubuntu/eviction_data/Eviction_Notices.csv')
# df_median_housingprice_2 = pd.read_csv('/home/ubuntu/eviction_data/med_sp_zip_code_sf_ca (1).csv')
# df_census = pd.read_csv('/home/ubuntu/eviction_data/ACS_data_total.csv')
# df_unemployment = pd.read_csv('/home/ubuntu/eviction_data/Unemployment_Rate.csv')

df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housingprice_2 = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')
#df_buyout = pd.read_csv('Buyout_agreements.csv')
#df_request = pd.read_csv('Requests_for_Information_Regarding_Protected_Status_Related_to_Owner_Move-In_Evictions.csv')


def data_processing_eviction(df_eviction):
    '''Performs a variety of transformations on eviction data from DataSF:

    PARAMETERS
    ----------
    df_eviction: Pandas DataFrame of eviction data from here:'''

    df_eviction_copy = df_eviction.copy()
    df_eviction_copy.replace(False,0,inplace=True)
    df_eviction_copy.replace(True,1,inplace=True)
    df_eviction_copy['Eviction_Notice'] = 1
    # df_eviction_copy = df_eviction_copy[(df_eviction_copy.Development==1)|\
    #                     (df_eviction_copy['Condo Conversion']==1)|\
    #                     (df_eviction_copy['Ellis Act WithDrawal']==1)|\
    #                     (df_eviction_copy['Substantial Rehab']==1)| \
    #                     (df_eviction_copy['Capital Improvement']==1)|\
    #                     (df_eviction_copy['Demolition']==1)|\
    #                     (df_eviction_copy['Owner Move In']==1)]
    df_eviction_copy['File Date'] = pd.to_datetime(df_eviction_copy['File Date'])
    df_eviction_copy['start_of_month_datetime'] = df_eviction_copy['File Date']-pd.offsets.MonthBegin(1)
    df_eviction_copy = df_eviction_copy.groupby(['start_of_month_datetime', 'Eviction Notice Source Zipcode']).sum().reset_index()
    df_eviction_copy['Year'] = df_eviction_copy['start_of_month_datetime'].dt.year
    df_eviction_copy['Month'] = df_eviction_copy['start_of_month_datetime'].dt.month
    return df_eviction_copy

def data_processing_housing(df_median_housingprice_2):
    '''Performs a variety of transformations on housing price data from Redfin.

    PARAMETERS
    ----------
    df_median_housing_price_2: Pandas data of median house prices in SF. From Redfin.
    '''

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
        'three_months_prior','six_months_prior','one_year_prior',\
        'median_sale_price_one_month_prior', 'median_sale_price_three_months_prior',\
        'median_sale_price_six_months_prior','median_sale_price_one_year_prior']]

    return df_median_housingprice_2

def merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment):
    '''Performs a variety of transformations and merges with eviction and housing price data

    PARAMETERS
    ----------
    df_eviction: Pandas DataFrame of eviction data from here:
    df_median_housing_price: Pandas data of median house prices in SF. From Redfin.
    df_census: aggregated data from the American Community Survey
    df_unemployment: unemployment percentage by month for the city of SF'''

    df_census['HC01_VC03']=df_census['HC01_VC03'].astype(int)
    df_census['GEO.id2']= df_census['GEO.id2'].astype(str)

    merged_df = pd.merge(df_eviction,df_median_housing_price,how='left',\
        left_on=['Eviction Notice Source Zipcode','start_of_month_datetime'],\
        right_on=['zip_code','period_begin'])


    merged_df = merged_df.drop(['median_sale_price'], axis=1)

    merged_df['Month_S'] = merged_df['Month'].map(int)
    merged_df['Year_S'] = merged_df['Year'].map(int)
    merged_df['Month_Year']= pd.to_datetime(dict(year=merged_df.Year_S, month=merged_df.Month_S, day=1))
    merged_df['one_month_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(1)

    merged_df = merged_df.dropna(subset = ['period_begin','median_sale_price_one_month_prior',\
        'median_sale_price_one_year_prior', 'median_sale_price_six_months_prior','median_sale_price_three_months_prior']).sort_values('period_begin')

    #
    merged_df = pd.merge(merged_df[['Month', 'Year', 'zip_code', 'median_sale_price_one_month_prior',\
        u'median_sale_price_three_months_prior',u'median_sale_price_six_months_prior',\
        u'median_sale_price_one_year_prior','Eviction_Notice', 'Month_Year', 'Month_S', 'Year_S','one_month_prior']],\
        merged_df[['Eviction_Notice','Month_Year','zip_code']], how='left', \
        left_on=['one_month_prior','zip_code'],right_on= ['Month_Year','zip_code'], suffixes=('','_one_month_prior'))
    #
    merged_df['datetime_one_year_prior']=merged_df['Month_Year']-pd.Timedelta(days=365)

    merged_df['one_year_prior']=merged_df['Year_S']-1
    merged_df['two_years_prior']=merged_df['Year_S']-2
    merged_df['three_years_prior']=merged_df['Year_S']-3
    merged_df['four_years_prior']=merged_df['Year_S']-4
    merged_df['five_years_prior']=merged_df['Year_S']-5
    merged_df['datetime_six_months_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(6)
    #
    merged_df = pd.merge(merged_df,merged_df[['Eviction_Notice','Month_Year','zip_code']], how='left', left_on=['one_month_prior','zip_code'],right_on= ['Month_Year','zip_code'], suffixes=('','_one_month_prior'))
    merged_df = pd.merge(merged_df,df_census[['GEO.id2','HC01_VC03','Year']],how='left',left_on=['zip_code', 'one_year_prior'], right_on=['GEO.id2','Year'], suffixes=('','_year_prior'))
    merged_df = pd.merge(merged_df,df_census[['GEO.id2','HC01_VC03','Year']],how='left',left_on=['zip_code', 'two_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_two_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['GEO.id2','HC01_VC03','Year']],how='left',left_on=['zip_code', 'three_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_three_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['GEO.id2','HC01_VC03','Year']],how='left',left_on=['zip_code', 'four_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_four_years_prior'))
    merged_df = pd.merge(merged_df,df_census[['GEO.id2','HC01_VC03','Year']],how='left',left_on=['zip_code', 'five_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_five_years_prior'))

    df_unemployment['DATE'] = pd.to_datetime(df_unemployment.DATE)
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_one_year_prior'], right_on=['DATE'], suffixes=('','_unemployment_year_prior'))
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_six_months_prior'], right_on=['DATE'], suffixes=('','_unemployment_six_months_prior'))

    return merged_df



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
    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_processed = data_processing_housing(df_median_housingprice_2)
    eviction_median_housing = merge_data(df_eviction_processed,df_median_housing_processed, df_census, df_unemployment)
