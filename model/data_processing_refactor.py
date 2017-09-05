
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
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')

def eviction_datetime_shift (df_eviction):

    df_eviction['File Date'] = pd.to_datetime(df_eviction['File Date'])
    df_eviction['start_of_month_datetime'] = df_eviction['File Date']-pd.offsets.MonthBegin(1)
    df_eviction['Address_Zipcode'] = df_eviction['Address_Zipcode'].apply(lambda x:0 if pd.isnull(x) else x)
    df_eviction['Address_Zipcode'] = df_eviction['Address_Zipcode'].apply(lambda x:0 if x=='Unknown_ZIP' else x)
    df_eviction['Address_Zipcode'] = df_eviction['Address_Zipcode'].astype(int).astype(str)
    df_eviction['Address_Zipcode'] = df_eviction['Address_Zipcode'].apply(lambda x:'Unknown_ZIP' if x=='0' else x)

    df_eviction = df_eviction.groupby(['start_of_month_datetime', 'Address_Zipcode']).sum().reset_index()
    df_eviction['Year'] = df_eviction['start_of_month_datetime'].dt.year
    df_eviction['Month'] = df_eviction['start_of_month_datetime'].dt.month

    return df_eviction

def eviction_boolean_cleaning (df_eviction):

    df_eviction.replace(False,0,inplace=True)
    df_eviction.replace(True,1,inplace=True)
    df_eviction['Supervisor District'] = df_eviction['Supervisor District'].astype(str)
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
    df_census['GEO.id2']= df_census['GEO.id2'].astype(str)

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
    merged_df['datetime_one_year_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(12)
    merged_df['datetime_two_years_prior']=merged_df['Month_Year']-pd.offsets.MonthBegin(24)

    return merged_df

def census_merge(merged_df, df_census):

    merged_df = pd.merge(merged_df,df_census,how='left',left_on=['Address_Zipcode', 'one_year_prior'], right_on=['GEO.id2','Year'], suffixes=('','_year_prior'))
    merged_df = pd.merge(merged_df,df_census,how='left',left_on=['Address_Zipcode', 'two_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_two_years_prior'))
    merged_df = pd.merge(merged_df,df_census,how='left',left_on=['Address_Zipcode', 'three_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_three_years_prior'))
    merged_df = pd.merge(merged_df,df_census,how='left',left_on=['Address_Zipcode', 'four_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_four_years_prior'))
    merged_df = pd.merge(merged_df,df_census,how='left',left_on=['Address_Zipcode', 'five_years_prior'], right_on=['GEO.id2','Year'], suffixes=('','_five_years_prior'))
    merged_df = merged_df.rename(columns={'Estimate; RACE - One race - Black or African American':'Black_population_previous_year','Percent; RACE - One race - White':'percent_white_population_previous_year'})

    return merged_df

def unemployment_merge(merged_df,df_unemployment):

    df_unemployment['DATE'] = pd.to_datetime(df_unemployment.DATE)
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_one_year_prior'], right_on=['DATE'], suffixes=('','_unemployment_year_prior'))
    merged_df = pd.merge(merged_df,df_unemployment ,how ='left', left_on=['datetime_six_months_prior'], right_on=['DATE'], suffixes=('','_unemployment_six_months_prior'))

    return merged_df

def housing_merge(df_eviction_processed,df_median_housing_price_processed):
    eviction_housing_merged_df = pd.merge(df_eviction_processed,df_median_housing_price_processed,how='left',\
        left_on=['Address_Zipcode','start_of_month_datetime'],\
        right_on=['zip_code','period_begin'])

    return eviction_housing_merged_df

def capital_improvement_merge(df,capital_improvement_filtered_df):
    merged = pd.merge(df,capital_improvement_filtered_df,how='left',left_on=['Address_Zipcode','datetime_one_year_prior'], right_on=['Petition Source Zipcode','Date Filed'])
    merged_2 = pd.merge(merged,capital_improvement_filtered_df,how='left',left_on=['Address_Zipcode','datetime_six_months_prior'], right_on=['Petition Source Zipcode','Date Filed'],suffixes=('','_six_months_prior'))
    merged_3 = pd.merge(merged_2,capital_improvement_filtered_df,how='left',left_on=['Address_Zipcode','datetime_two_years_prior'], right_on=['Petition Source Zipcode','Date Filed'],suffixes=('','_two_years_prior'))

    return merged_3

def capital_improvement_cleaning(df_capital_improvements):
    capital_improvement_filtered = df_capital_improvements[['Petition Source Zipcode','Date Filed','Ground (landlord): Capital Improvement']]
    capital_improvement_filtered['Date Filed'] = pd.to_datetime(capital_improvement_filtered['Date Filed']) - pd.offsets.MonthBegin(1)
    capital_improvement_group_by = capital_improvement_filtered.groupby(['Date Filed','Petition Source Zipcode']).sum().reset_index()
    capital_improvement_group_by = capital_improvement_group_by.rename(columns={'Ground (landlord): Capital Improvement':\
                                            'capital_improvement_petition',})
    return capital_improvement_group_by

def transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment):
    '''Performs a variety of transformations and merges with eviction and housing price data

    PARAMETERS
    ----------
    df_eviction: Pandas DataFrame of eviction data from here: https://data.sfgov.org/Housing-and-Buildings/Eviction-Notices/5cei-gny5
    df_median_housing_price: Pandas data of median house prices in SF. From Redfin.
    df_census: aggregated data from the American Community Survey: https://www.census.gov/programs-surveys/acs/
    df_unemployment: unemployment percentage by month for the city of SF: https://fred.stlouisfed.org/series/CASANF0URN'''

    #cleaning/datetime shifting/transformations of datasets
    cleaned_census_df = census_cleaning(df_census)
    capital_improvement_filtered_df = capital_improvement_cleaning(df_capital_improvements)
    df_eviction_processed = data_processing_eviction(df_eviction)
    df_median_housing_price_processed = data_processing_housing(df_median_housingprice)

    #merging datasets into single dataframe
    eviction_housing_merged_df = housing_merge(df_eviction_processed,df_median_housing_price_processed)
    datetime_shifted_df = merged_data_datetime_shift(eviction_housing_merged_df)
    census_merged_df = census_merge(datetime_shifted_df,cleaned_census_df)
    unemployment_merged_df= unemployment_merge(census_merged_df,df_unemployment)
    capital_improvement_merged_df = capital_improvement_merge(unemployment_merged_df, capital_improvement_filtered_df)
    final_merged_df = capital_improvement_merged_df

    return final_merged_df


def y_train_sort(df):
    params = {'94102': [(2, 1, 1), 532.5856587200159],
 '94103': [(0, 1, 1), 803.3662428898983],
 '94104': ['order', 40000000000],
 '94105': [(5, 0, 0), 13.658867630369574],
 '94107': [(0, 0, 0), 271.72564884092628],
 '94108': [(0, 0, 0), 309.45002369236113],
 '94109': [(1, 1, 1), 513.2303675590061],
 '94110': [(4, 1, 1), 558.9043335374527],
 '94111': [(0, 0, 0), 59.731675440435893],
 '94112': [(1, 1, 1), 489.46407667582207],
 '94114': [(0, 1, 1), 437.10399266503805],
 '94115': [(1, 1, 1), 425.69702383134984],
 '94116': [(0, 1, 1), 387.8427126258089],
 '94117': [(0, 1, 1), 484.74695270219246],
 '94118': [(0, 1, 1), 424.40096173225675],
 '94121': [(0, 1, 1), 442.91657962821625],
 '94122': [(2, 1, 1), 451.9743459843794],
 '94123': [(0, 0, 0), 385.11084487071133],
 '94124': [(0, 1, 1), 404.7179036313577],
 '94127': [(0, 0, 0), 193.42690896197888],
 '94131': [(2, 0, 1), 353.0419882704688],
 '94132': [(0, 1, 1), 768.1951226809797],
 '94133': [(0, 1, 1), 475.35291292581576],
 '94134': [(3, 1, 1), 363.41584411489987]}

    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','Year_S','Month_S','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day


    return sorted_2


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
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housingprice, df_census, df_unemployment)
