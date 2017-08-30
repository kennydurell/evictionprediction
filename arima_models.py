import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyflux as pf

import logging
import datetime

from data_processing_refactor import transform_merge_data
from cross_validation import zip_code_cv, arimax_by_zip_cv, arima_by_month_cv, arima_by_zip_data_transform, arima_by_zip_months
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
    zip_dict = {'94102': [(5, 1, 0), 623.266413592377],
             '94103': [(0, 1, 1), 947.777769286041],
             '94105': [(0, 0, 0), 40.180978167361289],
             '94107': [(0, 0, 0), 331.37020968909519],
             '94108': [(0, 1, 1), 364.6132573714833],
             '94109': [(0, 1, 1), 609.8987426017522],
             '94110': [(1, 1, 1), 672.7098595008938],
             '94111': [(0, 0, 0), 85.910297612209703],
             '94112': [(2, 1, 1), 618.0824554853265],
             '94114': [(0, 1, 1), 565.0623272048997],
             '94115': [(1, 1, 1), 499.83774848496137],
             '94116': [(0, 1, 1), 464.0206163588366],
             '94117': [(2, 1, 1), 595.2911496529726],
             '94118': [(0, 1, 1), 504.8016739606178],
             '94121': [(0, 1, 1), 528.2140349319886],
             '94122': [(0, 1, 1), 578.9831707056712],
             '94123': [(0, 1, 1), 455.47720806383063],
             '94124': [(0, 1, 1), 493.37090932980703],
             '94127': [(0, 0, 0), 250.66270841002387],
             '94131': [(0, 0, 0), 463.70946192544028],
             '94132': [(0, 1, 1), 843.3779121721473],
             '94133': [(0, 1, 1), 541.8146395991332],
             '94134': [(1, 1, 1), 471.9349755176075],
             '94158': [(0, 1, 0), 9.0264780133051765],
             'Unknown_ZIP': [(4, 1, 1), 20393.918844013817]}
    rmse_dict = {}
    i=0
    std_list = [1,2,3,4,6,10]

    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)

    error_counter=0
    forecasted_dict= {}
    #new_df = pd.DataFrame(columns=['zip_code','Date','true','forecasted'])
    true_dict = {}
    months_dict = {}

    for std in std_list:
        forecasted =defaultdict(list)
        true = defaultdict(list)
        months = defaultdict(list)
        for month in months_list:
            train_indices = transformed_df[transformed_df['Month_Year']<month].index
            test_indices = transformed_df[transformed_df['Month_Year']==month].index

            for zip_code,value in zip_dict.iteritems():
                y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
                if zip_code in y_train.Address_Zipcode.values and zip_code in y_test.Address_Zipcode.values:
                    try:
                        y_train = y_train[y_train['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN','percent_white_population_previous_year']]
                        y_test = y_test[y_test['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN','percent_white_population_previous_year']]
                        for i,row in y_train.Eviction_Notice.iteritems():
                            if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*std):
                                y_train.Eviction_Notice.iloc[i]=(y_train.Eviction_Notice.mean())

                        y_train = y_train.set_index(['Month_Year'], inplace=False)
                        y_test = y_test.set_index(['Month_Year'], inplace=False)

                        model = pf.ARIMAX(data=y_train, formula='Eviction_Notice~1+CASANF0URN',
                              ar=value[0][0], ma=value[0][2])
                        model.fit(exog=y_test)

                        y_hat = model.predict(h=1,oos_data=y_test)

                        if not np.isnan(np.asarray(y_hat)):
                            forecasted[zip_code].append(y_hat.Eviction_Notice.values.tolist())
                            true[zip_code].append(y_test.Eviction_Notice.values.tolist())
                            months[zip_code].append(month)
                            i+=1
                            print i
                    except Exception as e:
                        logging.error(e, exc_info=True)

        for zip_code in forecasted.keys():
            forecasted[zip_code]=[x for x in forecasted[zip_code]]
            true[zip_code]=[x for x in true[zip_code]]
            rmse =(mean_squared_error(np.asarray(true[zip_code]),np.asarray(forecasted[zip_code])))**.5

            if zip_code not in rmse_dict.keys():
                forecasted_dict[zip_code]= forecasted[zip_code]
                true_dict[zip_code]=true[zip_code]
                months_dict[zip_code]=months[zip_code]
                rmse_dict[zip_code]=[rmse,std]

            elif rmse<rmse_dict[zip_code][0]:
                forecasted_dict[zip_code]= forecasted[zip_code]
                true_dict[zip_code]=true[zip_code]
                months_dict[zip_code]=months[zip_code]
                rmse_dict[zip_code]=[rmse,std]

    return rmse_dict,true_dict, forecasted_dict, months_dict


def dataframe_transform(true_dict,forecasted_dict,months_dict):
    forecasted_dict[zip_code]
    pass

def historical_proportions(original_df,predictions_df):
    transformed_df = arima_by_zip_data_transform(original_df)

    percentage_of_month_df = pd.merge(original_df[['Month_Year','Address_Zipcode',\
    'Eviction_Notice']],predictions_df,how = 'left',left_on='Month_Year',right_on='month_year',\
    suffixes=('','_for_month'))
    percentage_of_month_df['perc_of_month']=percentage_of_month_df['Eviction_Notice']/percentage_of_month_df['actual_evictions']
    months_list = predictions_df[predictions_df.month_year>min(predictions_df.month_year)]['month_year']

    i=0
    zip_perc_df= pd.DataFrame(np.random.randn(1, 3), columns=['month_for_zips','Address_Zipcode','perc_of_month'])
    for month in months_list:
        indices = pd.notnull(percentage_of_month_df[percentage_of_month_df.Month_Year<month]['perc_of_month'])
        non_null_percentages_df = percentage_of_month_df[percentage_of_month_df.Month_Year<month].loc[indices]

        group_by_zip_df = non_null_percentages_df.groupby('Address_Zipcode').mean().reset_index()[['Address_Zipcode','perc_of_month']]
        #group_by_zip_df['predicted_eviction_for_zip'] = group_by_zip_df['perc_of_month']*predictions_df[predictions_df['month_year']==month]['predicted_evictions']
        group_by_zip_df['month_for_zips'] = month
        print group_by_zip_df
        zip_perc_df = zip_perc_df.append(group_by_zip_df, ignore_index=True)
        i+=1
        print i
    zip_perc_df = zip_perc_df[1:]
    zip_perc_df.month_for_zips = pd.to_datetime(zip_perc_df.month_for_zips)
    final_df = pd.merge(percentage_of_month_df,zip_perc_df,left_on=['Month_Year','Address_Zipcode'],right_on=['month_for_zips','Address_Zipcode'],suffixes=('','_averaged_over_past_months'))
    final_df['predicted_by_zip_evictions'] = final_df.perc_of_month_averaged_over_past_months*final_df.predicted_evictions

    return final_df


def arimax_overall (df):
    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)

    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()

    params = [(2, 1, 2), 36095.90415324242]
    forecasted =[]
    true = []
    months =[]
    i=0

    error_counter=0

    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[transformed_df['Month_Year']==month].index

        x_train, x_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
        y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
        try:
            x_train = x_train[['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year',]].set_index(['Month_Year'], inplace=False)
            x_test = x_test[['CASANF0URN','CASANF0URN_unemployment_six_months_prior','Month_Year']].set_index(['Month_Year'], inplace=False)
            for i,row in y_train.Eviction_Notice.iteritems():
                if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*2):
                    y_train.Eviction_Notice.iloc[i]=(sum(y_train.Eviction_Notice)-row)/(len(y_train.Eviction_Notice)-1)
            y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            y_test = y_test[['Month_Year','Eviction_Notice']]
            mod = ARIMA(endog=y_train,order=params[0], exog=x_train)
            results = mod.fit(exog=x_test)
            print x_train
            print x_test
            print y_test
            break
            y_hat = results.forecast()[0].tolist()

            if not np.isnan(np.asarray(y_hat)):
                forecasted.append(y_hat)
                true.append(y_test.Eviction_Notice.values.tolist())
                months.append(month)
                i+=1
                print i
        except Exception as e:
            logging.error(e, exc_info=True)
    forecasted=[y for x in forecasted for y in x]
    true=[y for x in true for y in x]
    rmse =(mean_squared_error(np.asarray(true),np.asarray(forecasted)))**.5
    return rmse, np.asarray(true), np.asarray(forecasted), months



def pyflux_overall (df):
    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)

    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()

    params = [(2, 1, 2), 36095.90415324242]
    forecasted =[]
    true = []
    months =[]
    i=0


    error_counter=0

    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[transformed_df['Month_Year']==month].index

        y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
        try:
            for i,row in y_train.Eviction_Notice.iteritems():
                if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*3):
                    y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()
            y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            y_test = y_test[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            model = pf.ARIMA(data=y_train,ar=2, ma=2)
            model.fit()

            y_hat = model.predict(h=1)
            print 'y_test:', y_test
            print 'y_hat', y_hat
            if not np.isnan(np.asarray(y_hat)):
                forecasted.append(y_hat.Eviction_Notice.tolist())
                true.append(y_test.Eviction_Notice.values.tolist())
                months.append(month)
                i+=1
                print i
            #    print 'y_test:', y_test
            #    print 'y_hat', y_hat
        except Exception as e:
            logging.error(e, exc_info=True)

    forecasted=[y for x in forecasted for y in x]
    true=[y for x in true for y in x]
    rmse =(mean_squared_error(np.asarray(true),np.asarray(forecasted)))**.5
    return rmse, np.asarray(true), np.asarray(forecasted), months


def arimax_by_month (df):
    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)
    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()
    params = [(2, 1, 2), 36095.90415324242]
    predictions_df= pd.DataFrame(np.random.randn(1, 3), columns=['month_year', 'actual_evictions', 'predicted_evictions'])
    i=0


    error_counter=0

    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[transformed_df['Month_Year']==month].index

        y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
        try:
            for i,row in y_train.Eviction_Notice.iteritems():
                if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*3):
                    y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()
            y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            y_test = y_test[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
            model = pf.ARIMA(data=y_train,ar=2, ma=2)
            model.fit()

            y_hat = model.predict(h=1)

            if not np.isnan(np.asarray(y_hat)):
                df_values = [month] + y_test.Eviction_Notice.tolist() + y_hat.Eviction_Notice.tolist()
                temp_df = pd.DataFrame([df_values],columns=['month_year','actual_evictions','predicted_evictions'])
                predictions_df = predictions_df.append(temp_df, ignore_index=True)
                i+=1
                print i
            #    print 'y_test:', y_test
            #    print 'y_hat', y_hat
        except Exception as e:
            logging.error(e, exc_info=True)

    rmse =(mean_squared_error(predictions_df[1:].actual_evictions,predictions_df[1:].predicted_evictions))**.5
    predictions_df = predictions_df[1:]
    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])
    return rmse, predictions_df


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






if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    #rmse_dict, true, forecasted, months = arimax_by_zip(eviction_median_housing)
    #plot_models(rmse_dict,true,forecasted,months)
    #rmse, true, forecasted, months = arimax_overall(eviction_median_housing)
    #rmse, true, forecasted, months = pyflux_overall(eviction_median_housing)
    rmse, predictions_by_month_df= arimax_by_month(eviction_median_housing)
    zip_perc_df = historical_proportions(eviction_median_housing,predictions_by_month_df)

    #zip_code_best = zip_code_cv(eviction_median_housing)
    # results, pdq_best, seasonal_best = param_check_overall(eviction_median_housing)
    # print 'Best sarima model for each zip'
    # print zip_code_best
    # print 'Best sarima model for total evictions'
    # print results,pdq_best,seasonal_best
    #rmse, true, forecasted, months_list = arimax_overall(eviction_median_housing)
