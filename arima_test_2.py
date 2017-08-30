import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm

import datetime

from data_processing_refactor import transform_merge_data
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict

# Modelling Algorithms

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

#For use on EC2 instances
# df_eviction = pd.read_csv('/home/ubuntu/eviction_data/Eviction_Notices.csv')
# df_median_housingprice_2 = pd.read_csv('/home/ubuntu/eviction_data/med_sp_zip_code_sf_ca (1).csv')
# df_census = pd.read_csv('/home/ubuntu/eviction_data/ACS_data_total.csv')
# df_unemployment = pd.read_csv('/home/ubuntu/eviction_data/Unemployment_Rate.csv')
#

#importing files
df_eviction = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Eviction_Notices.csv')
df_median_housing_price = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/med_sp_zip_code_sf_ca (1).csv')
df_census = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/ACS_data_total.csv')
df_unemployment = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Unemployment_Rate.csv')



def zip_code_cv(df):
    zip_code_arima_best =defaultdict(list)
    i=0
    for zip_code in df['zip_code'].unique():
        results_list = arima_by_zip_cv(df, zip_code)
        zip_code_arima_best[zip_code] = results_list
        i+=1
        print i
    return zip_code_arima_best


def arima_by_zip_cv (df,zip_code):

    p = range(0,6)
    d= range(0,2)
    q= range(0,2)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    sorted_df = df[['Month_Year','Eviction_Notice','zip_code']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)

    aic_list = ['param',400000000]
    i=0

    y_train = sorted_df[sorted_df['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    for param in pdq:
        try:
            mod = ARIMA(endog=y_train,order=param)
            results = mod.fit()
            if results.aic<aic_list[1]:
                aic_list=[param,results.aic]
            i+=1
            print i
        except:
            continue

    return aic_list

def arima_by_month_cv (df):
    p = range(0,6)
    d= range(0,3)
    q= range(1,3)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    sorted_df = df[['Month_Year','Eviction_Notice']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)
    y_train=sorted_df.set_index(['Month_Year'], inplace=False)

    aic_list = ['param',400000000]
    i=0
    for param in pdq:
        try:
            mod = ARIMA(endog=y_train,order=param)
            results = mod.fit()
            if results.aic<aic_list[1]:
                aic_list=[param,results.aic]
            i+=1
            print i
        except:
            continue
    return aic_list


def sarimax (df):
    plt.gcf().clear()
    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2.dropna(subset=['CASANF0URN','CASANF0URN_unemployment_six_months_prior'], inplace=True)
    #list_of_zips = np.sort(sorted_2.zip_code.unique())
    zip_dict = {'94102': [(2, 1, 1), 532.5856587200159],
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
    rmse_dict = {}
    i=0
    for zip_code,value in zip_dict.iteritems():
        #x = sorted_2[sorted_2['zip_code']==zip_code][['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year']].set_index(['Month_Year'], inplace=False)
        # x=sm.add_constant(x)
        y = sorted_2[sorted_2['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
        print zip_code
        #
        if '2011-01-01' in y.index and zip_code!='94127':
            mod = SARIMAX(endog=y,
                                    order=value[0],
                                    seasonal_order=value[1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)


            results = mod.fit()
            pred = results.get_prediction(start=pd.to_datetime('2011-01-01'), dynamic=False)
            pred_ci = pred.conf_int()
            ax = y['2011':].plot(label='observed')
            pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
            # ax.fill_between(pred_ci.index,
            #         pred_ci.iloc[:, 0],
            #         pred_ci.iloc[:, 1], color='k', alpha=.2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Evictions')
            ax.set_title(zip_code)
            plt.legend()


            y_forecasted = pred.predicted_mean
            y_truth = y['2011-01-01':]
            rmse = (((y_forecasted.values - y_truth.values) ** 2).mean())**.5
            rmse_dict[zip_code]=rmse
            i+=1
            print i
            print zip_code
            print results.summary()
    return rmse_dict


def arimax_by_zip (df):
    plt.gcf().clear()
    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','Year_S','Month_S','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day
    sorted_2.dropna(subset=['CASANF0URN','CASANF0URN_unemployment_six_months_prior'], inplace=True)
    sorted_2.reset_index(inplace=True)
    zip_dict = {'94102': [(2, 1, 1), 532.5856587200159],
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
    rmse_dict = {'94102': [40000000, 'none'],
 '94103': [40000000, 'none'],
 '94107': [40000000, 'none'],
 '94108': [40000000, 'none'],
 '94109': [40000000, 'none'],
 '94110': [40000000, 'none'],
 '94111': [40000000, 'none'],
 '94112': [40000000, 'none'],
 '94114': [40000000, 'none'],
 '94115': [40000000, 'none'],
 '94116': [40000000, 'none'],
 '94117': [40000000, 'none'],
 '94118': [40000000, 'none'],
 '94121': [40000000, 'none'],
 '94122': [40000000, 'none'],
 '94123': [40000000, 'none'],
 '94124': [40000000, 'none'],
 '94127': [40000000, 'none'],
 '94131': [40000000, 'none'],
 '94132': [40000000, 'none'],
 '94133': [40000000, 'none'],
 '94134': [40000000, 'none']}

    i=0
    std_list = [1,2,3,4,6,10]

    months = sorted_2[sorted_2['Month_Year']>(min(sorted_2['Month_Year'])+pd.offsets.MonthBegin(3))][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]
    error_counter=0
    forecasted_dict= {}
    new_df = pd.DataFrame(columns=['zip_code','Date','true','forecasted'])
    true_dict = {}
    months_dict = {}
    for std in std_list:
        forecasted =defaultdict(list)
        true = defaultdict(list)
        months = defaultdict(list)
        for month in months_list:
            train_indices = sorted_2[sorted_2['Month_Year']<month].index
            test_indices = sorted_2[sorted_2['Month_Year']==month].index

            for zip_code,value in zip_dict.iteritems():
                x_train = sorted_2.iloc[train_indices]
                y_train, y_test = sorted_2.iloc[train_indices], sorted_2.iloc[test_indices]
                if zip_code in y_train.zip_code.values and zip_code in y_test.zip_code.values:
                    try:
                        x_train = x_train[x_train['zip_code']==zip_code][['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year']].set_index(['Month_Year'], inplace=False)
                        y_train = y_train[y_train['zip_code']==zip_code][['Month_Year','Eviction_Notice']]
                        y_test = y_test[y_test['zip_code']==zip_code][['Month_Year','Eviction_Notice']]
                        for i,row in y_train.Eviction_Notice.iteritems():
                            if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*std):
                                y_train.Eviction_Notice.iloc[i]=(y_train.Eviction_Notice.mean())

                        y_train = y_train.set_index(['Month_Year'], inplace=False)
                        y_test = y_test.set_index(['Month_Year'], inplace=False)

                        mod = ARIMA(endog=y_train,order=value[0])

                        results = mod.fit()
                        y_hat = results.forecast()[0].tolist()

                        if not np.isnan(np.asarray(y_hat)):
                            forecasted[zip_code].append(y_hat)
                            true[zip_code].append(y_test.Eviction_Notice.values.tolist())
                            months[zip_code].append(month)
                            i+=1
                            print i
                    except:
                        error_counter+=1

        for zip_code in forecasted.keys():
            forecasted[zip_code]=[x for x in forecasted[zip_code]]
            true[zip_code]=[x for x in true[zip_code]]
            rmse =(mean_squared_error(np.asarray(true[zip_code]),np.asarray(forecasted[zip_code])))**.5
            if rmse<rmse_dict[zip_code][0]:
                forecasted_dict[zip_code]= forecasted[zip_code]
                true_dict[zip_code]=true[zip_code]
                months_dict[zip_code]=months[zip_code]
                rmse_dict[zip_code]=[rmse,std]

    return rmse_dict,true_dict, forecasted_dict, months_dict


def dataframe_transform(true_dict,forecasted_dict,months_dict):
    forecasted_dict[zip_code]

def arimax_overall (df):
    plt.gcf().clear()
    sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','Year_S','Month_S','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
    sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
    sorted_2['Day_S'] = sorted_2['Month_Year'].dt.day
    sorted_2.dropna(subset=['CASANF0URN','CASANF0URN_unemployment_six_months_prior'], inplace=True)

    months = sorted_2[sorted_2['Month_Year']>(min(sorted_2['Month_Year'])+pd.offsets.MonthBegin(1))][['Year_S','Month_S','Day_S']]
    months.drop_duplicates(inplace=True)
    months_list = [datetime.datetime(*x) for x in months.values]

    sorted_2=sorted_2.groupby('Month_Year').sum().reset_index()

    params = arima_by_month_cv(df)
    forecasted =[]
    true = []
    months =[]
    i=0


    error_counter=0

    for month in months_list:
        train_indices = sorted_2[sorted_2['Month_Year']<month].index
        test_indices = sorted_2[sorted_2['Month_Year']==month].index

        #x_train = sorted_2.iloc[train_indices]
        y_train, y_test = sorted_2.iloc[train_indices], sorted_2.iloc[test_indices]
        try:
            #x_train = x_train[['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year']].set_index(['Month_Year'], inplace=False)
            y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            y_test = y_test[['Month_Year','Eviction_Notice']]
            mod = ARIMA(endog=y_train,order=params[0])

            results = mod.fit()
            y_hat = results.forecast()[0].tolist()
            if not np.isnan(np.asarray(y_hat)):
                forecasted.append(y_hat)
                true.append(y_test.Eviction_Notice.values.tolist())
                months.append(month)
                i+=1
                print i
        except:
            error_counter+=1
    forecasted=[y for x in forecasted for y in x]
    true=[y for x in true for y in x]
    rmse =(mean_squared_error(np.asarray(true),np.asarray(forecasted)))**.5
    return rmse, np.asarray(true), np.asarray(forecasted), months


def plot_predictions(true_array,forecasted_array, months_array,zip_code=None):

    if zip_code != None:
        ax = true_array[zip_code].plot(label='observed')
        plt.plot(months_array[zip_code],true_array[zip_code])
        plt.plot(months_array[zip_code],forecasted_array[zip_code])
        plt.legend()
    else:
        plt.plot(months_array,true_array)
        plt.plot(months_array, forecasted_array)


def plot_models (rmse,true,forecasted,months):
    if type(rmse_dict) is dict:
        for zip_code in rmse_dict.keys():
            plot_predictions(true,forecasted,months,zip_code)
    else:
        plot_predictions(true,forecasted,months)


def param_check_overall (df):
    p = range(0,4)
    d= range(0,2)
    q= range(0,2)
    s = [6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

    #reformatting the data for use in SARIMAX model
    sorted_df = df[['Month_Year','Eviction_Notice']].groupby('Month_Year').sum().reset_index()
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)
    y = sorted_df[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    results = 40000000
    pdq_best = None
    seasonal_best = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(y,order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                if results.aic<results:
                    results = results.aic
                    pdq_best = param
                    seasonal_best = param_seasonal
            except:
                continue
    return results, pdq_best, seasonal_best


def param_check_by_zip (df, zip_code):
    p = range(0,4)
    d= range(0,2)
    q= range(0,2)
    s = [6,12]

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q, s))]

    #reformatting the data for use in SARIMAX model
    sorted_df = df[['Month_Year','Eviction_Notice','zip_code']].sort_values(['Month_Year'])
    sorted_df['Eviction_Notice']= sorted_df['Eviction_Notice'].astype(float)

    y = sorted_df[sorted_df['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)

    results = 40000000
    pdq_best = None
    seasonal_best = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(y,order=param, seasonal_order=param_seasonal,enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                if results.aic<results:
                    results = results.aic
                    pdq_best = param
                    seasonal_best = param_seasonal
            except:
                continue
    return results, pdq_best, seasonal_best






if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    rmse_dict, true, forecasted, months = arimax_by_zip(eviction_median_housing)
    plot_models(rmse_dict,true,forecasted,months)



    #zip_code_best = zip_code_cv(eviction_median_housing)
    # results, pdq_best, seasonal_best = param_check_overall(eviction_median_housing)
    # print 'Best sarima model for each zip'
    # print zip_code_best
    # print 'Best sarima model for total evictions'
    # print results,pdq_best,seasonal_best
    #rmse, true, forecasted, months_list = arimax_overall(eviction_median_housing)
