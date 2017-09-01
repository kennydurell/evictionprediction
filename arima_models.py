import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyflux as pf

import logging
import datetime

from data_processing_refactor import transform_merge_data
from cross_validation import arima_by_zip_data_transform, arima_by_zip_months
from pandas.tools.plotting import autocorrelation_plot

from collections import defaultdict

# Modelling Algorithms

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
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
df_capital_improvements = pd.read_csv('/Users/mightyhive/Desktop/Galvanize_Course/evictionprediction/Eviction_Data/Petitions_to_the_Rent_Board.csv')

def arimax_by_zip (df,zip_param_dictionary):


    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)

    predictions_df= pd.DataFrame(np.random.randn(1, 4), columns=['month_year', 'zip_code','actual_evictions', 'predicted_evictions'])

    i_2=0
    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[transformed_df['Month_Year']==month].index

        for zip_code,value in zip_param_dictionary.iteritems():
            y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
            if zip_code in y_train.Address_Zipcode.values and zip_code in y_test.Address_Zipcode.values:
                y_train = y_train[y_train['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN']].reset_index(inplace=False)
                y_test = y_test[y_test['Address_Zipcode']==zip_code][['Month_Year','Eviction_Notice','CASANF0URN']].reset_index(inplace=False)

                for i,row in y_train.Eviction_Notice.iteritems():
                    if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*value[1]):
                        y_train.Eviction_Notice.iloc[i]=(y_train.Eviction_Notice.mean())


                y_train = y_train.set_index(['Month_Year'], inplace=False)
                y_test = y_test.set_index(['Month_Year'], inplace=False)

                try:
                    model = pf.ARIMAX(data=y_train, formula='Eviction_Notice~1+CASANF0URN',
                          ar=value[0][0], ma=value[0][2])
                    model.fit()
                    y_hat = model.predict(h=1,oos_data=y_test)
                    predicted = y_hat.Eviction_Notice.values.tolist()
                    actual = y_test.Eviction_Notice.values.tolist()

                    temp_df = pd.DataFrame(data={'predicted_evictions':predicted,'actual_evictions':actual,\
                                            'zip_code': zip_code})
                    temp_df['month_year']=month
                    #print temp_df
                    predictions_df = predictions_df.append(temp_df,ignore_index=True)
                    i_2+=1
                    print i_2

                except Exception as e:
                    logging.error(e, exc_info=True)

    predictions_df=predictions_df[1:]
    predictions_df.month_year = pd.to_datetime(predictions_df.month_year)

    rmse_dict = {}
    for zip_code in predictions_df.zip_code.unique():
        zip_filtered_df = predictions_df[predictions_df.zip_code==zip_code]
        rmse_dict[zip_code] = (mean_squared_error(zip_filtered_df['actual_evictions'],zip_filtered_df['predicted_evictions']))**0.5


    return predictions_df, rmse_dict


def dataframe_transform(true_dict,forecasted_dict,months_dict):
    forecasted_dict[zip_code]
    pass

def top_down_estimation_by_zip(original_df):
    predictions_by_month_df, rmse= arimax_by_month(eviction_median_housing)

    percentage_of_month_df = pd.merge(original_df[['Month_Year','Address_Zipcode',\
    'Eviction_Notice']],predictions_by_month_df,how = 'left',left_on='Month_Year',right_on='month_year',\
    suffixes=('','_for_month'))
    percentage_of_month_df['perc_of_month']=percentage_of_month_df['Eviction_Notice']/percentage_of_month_df['actual_evictions']
    months_list = predictions_by_month_df[predictions_by_month_df.month_year>min(predictions_by_month_df.month_year)]['month_year']

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
    top_down_by_zip_df = final_df

    return top_down_by_zip_df


def arimax_by_month (df):

    transformed_df = arima_by_zip_data_transform(df)
    months_list = arima_by_zip_months(transformed_df)
    transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()
    params = [(2, 1, 2), 36095.90415324242]
    predictions_df= pd.DataFrame(np.random.randn(1, 3), columns=['month_year', 'actual_evictions', 'predicted_evictions'])


    i=0
    for month in months_list:
        train_indices = transformed_df[transformed_df['Month_Year']<month].index
        test_indices = transformed_df[(transformed_df['Month_Year']==month)|\
        (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(1)))|\
        (transformed_df['Month_Year']==(month+pd.offsets.MonthBegin(2)))].index

        y_train, y_test = transformed_df.iloc[train_indices].reset_index(inplace=False), transformed_df.iloc[test_indices].reset_index(inplace=False)
        for i,row in y_train.Eviction_Notice.iteritems():
            if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*1):
                y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()

        y_train = y_train[['Month_Year','Eviction_Notice','CASANF0URN']].set_index(['Month_Year'], inplace=False)
        y_test = y_test[['Month_Year','Eviction_Notice','CASANF0URN']].set_index(['Month_Year'], inplace=False)

        try:
            model = pf.ARIMAX(data=y_train,formula='Eviction_Notice~1+CASANF0URN',ar=2, ma=2)
            model.fit()
            y_hat = model.predict(h=3,oos_data=y_test)
            temp_df = pd.DataFrame({'actual_evictions':y_test.Eviction_Notice.values.tolist(),\
                                        'predicted_evictions':y_hat.Eviction_Notice.values.tolist(),'months_ahead':[1,2,3]})
            temp_df['month_year']=pd.Series([month,month+pd.offsets.MonthBegin(1),month+pd.offsets.MonthBegin(2)]).values
            predictions_df = predictions_df.append(temp_df, ignore_index=True)
            i+=1
            print i

        except Exception as e:
            logging.error(e, exc_info=True)

    # rmse =(mean_squared_error(predictions_df[1:].actual_evictions,predictions_df[1:].predicted_evictions))**.5
    predictions_df = predictions_df[1:]
    predictions_df['month_year']=pd.to_datetime(predictions_df['month_year'])
    return predictions_df


def zip_level_plot(predictions_df,zip_code,columns_to_plot=['Month_Year','actual_evictions',\
                                        'linear_combination_predicted_evictions','zip_code']):

    zip_filtered_df = predictions_df[predictions_df[columns_to_plot[3]]==zip_code]

    fig,ax = plt.subplots()
    ax.plot(zip_filtered_df[columns_to_plot[0]],zip_filtered_df[columns_to_plot[1]],label='Actual Eviction Notices')
    ax.plot(zip_filtered_df[columns_to_plot[0]],zip_filtered_df[columns_to_plot[2]],label='Predicted Eviction Notices')

    ax.set(title='Actual and Predicted Eviction Notices in SF ZIP code '+ str(zip_code) + ' (ARIMAX)',\
            ylabel = 'Evictions', xlabel='Month of Eviction Notice')
    ax.legend()


def entire_city_plot(predictions_df,columns_to_plot=['month_year','actual_evictions',\
                                        'predicted_evictions','zip_code']):
    fig,ax = plt.subplots()
    ax.plot(predictions_df[columns_to_plot[0]],predictions_df[columns_to_plot[1]],\
                                                    label='Actual Eviction Notices')
    ax.plot(predictions_df[columns_to_plot[0]],predictions_df[columns_to_plot[2]],\
                                                    label='Predicted Eviction Notices')

    ax.set(title='Actual and Predicted Eviction Notices in San Francisco (ARIMAX)',\
                            ylabel = 'Evictions', xlabel='Month of Eviction Notice')
    ax.legend()


def plot_models (df,zip_code=False):

    if zip_code:
        for zip_code_1 in df.zip_code.unique():
            zip_level_plot(df,zip_code=zip_code_1)
    else:
        entire_city_plot(predictions_df)



if __name__ == '__main__':
    eviction_median_housing = transform_merge_data(df_eviction,df_median_housing_price, df_census, df_unemployment)
    zip_dict = {'94102': [(5, 1, 0),3],
                '94103': [(0, 1, 1),3],
                 '94105': [(0, 0, 0),3],
                 '94107': [(0, 0, 0),3],
                 '94108': [(0, 1, 1),3],
                 '94109': [(0, 1, 1),3],
                 '94110': [(1, 1, 1),3],
                 '94111': [(0, 0, 0),3],
                 '94112': [(2, 1, 1),3],
                 '94114': [(0, 1, 1),3],
                 '94115': [(1, 1, 1),3],
                 '94116': [(0, 1, 1),3],
                 '94117': [(2, 1, 1),3],
                 '94118': [(0, 1, 1),3],
                 '94121': [(0, 1, 1),3],
                 '94122': [(0, 1, 1),3],
                 '94123': [(0, 1, 1),3],
                 '94124': [(0, 1, 1),3],
                 '94127': [(0, 0, 0),3],
                 '94131': [(0, 0, 0),3],
                 '94132': [(0, 1, 1),3],
                 '94133': [(0, 1, 1),3],
                 '94134': [(1, 1, 1),3],
                 '94158': [(0, 1, 0),3],
                 'Unknown_ZIP': [(4, 1, 1),3]}
    #arimax_by_zip_df,rmse_dict = arimax_by_zip(eviction_median_housing,zip_dict)
    #plot_models(rmse_dict,true,forecasted,months)
    #rmse, true, forecasted, months = arimax_overall(eviction_median_housing)
    #rmse, true, forecasted, months = pyflux_overall(eviction_median_housing)
    predictions_by_month_df= arimax_by_month(eviction_median_housing)
    #top_down_by_zip_df = top_down_estimation_by_zip(eviction_median_housing)
    #plot_models(zip_perc_df,zip_code=True)
    #plot_predictions(predictions_by_month_df)
    #zip_code_best = zip_code_cv(eviction_median_housing)
    # results, pdq_best, seasonal_best = param_check_overall(eviction_median_housing)
    # print 'Best sarima model for each zip'
    # print zip_code_best
    # print 'Best sarima model for total evictions'
    # print results,pdq_best,seasonal_best
    #rmse, true, forecasted, months_list = arimax_overall(eviction_median_housing)



    # def arimax_overall (df):
    #     transformed_df = arima_by_zip_data_transform(df)
    #     months_list = arima_by_zip_months(transformed_df)
    #
    #     transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()
    #
    #     params = [(2, 1, 2), 36095.90415324242]
    #     forecasted =[]
    #     true = []
    #     months =[]
    #     i=0
    #
    #     error_counter=0
    #
    #     for month in months_list:
    #         train_indices = transformed_df[transformed_df['Month_Year']<month].index
    #         test_indices = transformed_df[transformed_df['Month_Year']==month].index
    #
    #         x_train, x_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
    #         y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
    #         try:
    #             x_train = x_train[['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year',]].set_index(['Month_Year'], inplace=False)
    #             x_test = x_test[['CASANF0URN','CASANF0URN_unemployment_six_months_prior','Month_Year']].set_index(['Month_Year'], inplace=False)
    #             for i,row in y_train.Eviction_Notice.iteritems():
    #                 if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*2):
    #                     y_train.Eviction_Notice.iloc[i]=(sum(y_train.Eviction_Notice)-row)/(len(y_train.Eviction_Notice)-1)
    #             y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
    #             y_test = y_test[['Month_Year','Eviction_Notice']]
    #             mod = ARIMA(endog=y_train,order=params[0], exog=x_train)
    #             results = mod.fit(exog=x_test)
    #             print x_train
    #             print x_test
    #             print y_test
    #             break
    #             y_hat = results.forecast()[0].tolist()
    #
    #             if not np.isnan(np.asarray(y_hat)):
    #                 forecasted.append(y_hat)
    #                 true.append(y_test.Eviction_Notice.values.tolist())
    #                 months.append(month)
    #                 i+=1
    #                 print i
    #         except Exception as e:
    #             logging.error(e, exc_info=True)
    #     forecasted=[y for x in forecasted for y in x]
    #     true=[y for x in true for y in x]
    #     rmse =(mean_squared_error(np.asarray(true),np.asarray(forecasted)))**.5
    #     return rmse, np.asarray(true), np.asarray(forecasted), months
    #
    #
    #
    # def pyflux_overall (df):
    #     transformed_df = arima_by_zip_data_transform(df)
    #     months_list = arima_by_zip_months(transformed_df)
    #
    #     transformed_df=transformed_df.groupby('Month_Year').sum().reset_index()
    #
    #     params = [(2, 1, 2), 36095.90415324242]
    #     forecasted =[]
    #     true = []
    #     months =[]
    #     i=0
    #
    #
    #     error_counter=0
    #
    #     for month in months_list:
    #         train_indices = transformed_df[transformed_df['Month_Year']<month].index
    #         test_indices = transformed_df[transformed_df['Month_Year']==month].index
    #
    #         y_train, y_test = transformed_df.iloc[train_indices], transformed_df.iloc[test_indices]
    #         try:
    #             for i,row in y_train.Eviction_Notice.iteritems():
    #                 if row>(y_train.Eviction_Notice.mean()+ y_train.Eviction_Notice.std()*3):
    #                     y_train.Eviction_Notice.iloc[i]=y_train.Eviction_Notice.mean()
    #             y_train = y_train[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
    #             y_test = y_test[['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
    #             model = pf.ARIMA(data=y_train,ar=2, ma=2)
    #             model.fit()
    #
    #             y_hat = model.predict(h=1)
    #             print 'y_test:', y_test
    #             print 'y_hat', y_hat
    #             if not np.isnan(np.asarray(y_hat)):
    #                 forecasted.append(y_hat.Eviction_Notice.tolist())
    #                 true.append(y_test.Eviction_Notice.values.tolist())
    #                 months.append(month)
    #                 i+=1
    #                 print i
    #             #    print 'y_test:', y_test
    #             #    print 'y_hat', y_hat
    #         except Exception as e:
    #             logging.error(e, exc_info=True)
    #
    #     forecasted=[y for x in forecasted for y in x]
    #     true=[y for x in true for y in x]
    #     rmse =(mean_squared_error(np.asarray(true),np.asarray(forecasted)))**.5
    #     return rmse, np.asarray(true), np.asarray(forecasted), months


# def sarimax (df):
#     plt.gcf().clear()
#     sorted_2 = df[['Month_Year','Eviction_Notice','zip_code','CASANF0URN','CASANF0URN_unemployment_six_months_prior']].sort_values(['Month_Year'])
#     sorted_2['Eviction_Notice']= sorted_2['Eviction_Notice'].astype(float)
#     sorted_2.dropna(subset=['CASANF0URN','CASANF0URN_unemployment_six_months_prior'], inplace=True)
#     #list_of_zips = np.sort(sorted_2.zip_code.unique())
#     zip_dict = {'94102': [(2, 1, 1), 532.5856587200159],
#  '94103': [(0, 1, 1), 803.3662428898983],
#  '94104': ['order', 40000000000],
#  '94105': [(5, 0, 0), 13.658867630369574],
#  '94107': [(0, 0, 0), 271.72564884092628],
#  '94108': [(0, 0, 0), 309.45002369236113],
#  '94109': [(1, 1, 1), 513.2303675590061],
#  '94110': [(4, 1, 1), 558.9043335374527],
#  '94111': [(0, 0, 0), 59.731675440435893],
#  '94112': [(1, 1, 1), 489.46407667582207],
#  '94114': [(0, 1, 1), 437.10399266503805],
#  '94115': [(1, 1, 1), 425.69702383134984],
#  '94116': [(0, 1, 1), 387.8427126258089],
#  '94117': [(0, 1, 1), 484.74695270219246],
#  '94118': [(0, 1, 1), 424.40096173225675],
#  '94121': [(0, 1, 1), 442.91657962821625],
#  '94122': [(2, 1, 1), 451.9743459843794],
#  '94123': [(0, 0, 0), 385.11084487071133],
#  '94124': [(0, 1, 1), 404.7179036313577],
#  '94127': [(0, 0, 0), 193.42690896197888],
#  '94131': [(2, 0, 1), 353.0419882704688],
#  '94132': [(0, 1, 1), 768.1951226809797],
#  '94133': [(0, 1, 1), 475.35291292581576],
#  '94134': [(3, 1, 1), 363.41584411489987]}
#     rmse_dict = {}
#     i=0
#     for zip_code,value in zip_dict.iteritems():
#         #x = sorted_2[sorted_2['zip_code']==zip_code][['CASANF0URN','CASANF0URN_unemployment_six_months_prior', 'Month_Year']].set_index(['Month_Year'], inplace=False)
#         # x=sm.add_constant(x)
#         y = sorted_2[sorted_2['zip_code']==zip_code][['Month_Year','Eviction_Notice']].set_index(['Month_Year'], inplace=False)
#         print zip_code
#         #
#         if '2011-01-01' in y.index and zip_code!='94127':
#             mod = SARIMAX(endog=y,
#                                     order=value[0],
#                                     seasonal_order=value[1],
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#
#
#             results = mod.fit()
#             pred = results.get_prediction(start=pd.to_datetime('2011-01-01'), dynamic=False)
#             pred_ci = pred.conf_int()
#             ax = y['2011':].plot(label='observed')
#             pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
#             # ax.fill_between(pred_ci.index,
#             #         pred_ci.iloc[:, 0],
#             #         pred_ci.iloc[:, 1], color='k', alpha=.2)
#             ax.set_xlabel('Date')
#             ax.set_ylabel('Total Evictions')
#             ax.set_title(zip_code)
#             plt.legend()
#
#
#             y_forecasted = pred.predicted_mean
#             y_truth = y['2011-01-01':]
#             rmse = (((y_forecasted.values - y_truth.values) ** 2).mean())**.5
#             rmse_dict[zip_code]=rmse
#             i+=1
#             print i
#             print zip_code
#             print results.summary()
#     return rmse_dict
