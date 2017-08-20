import numpy as np
import pandas as pd
import datetime
from data_processing import data_processing_eviction, data_processing_housing, merge_data

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


def Model(df):
    y_true_values = defaultdict(list)
    y_predicted_values = defaultdict(list)
    rfr = RandomForestRegressor()
    list_of_zips = df.zip_code.unique()

    for zip_code in list_of_zips:
        zip_filtered = df[df['zip_code']==zip_code].sort_values(['Year','Month'])
        y = zip_filtered.pop('Eviction_Notice')
        X = zip_filtered
        mse_by_zip = defaultdict(int)

        if len(X.index)>10:

            tscv = TimeSeriesSplit(n_splits=len(X.index)-1)

            for train_index, test_index in tscv.split(X):
            #     print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index].values
                rfr.fit(X_train,y_train)
                y_hat = rfr.predict(X_test).tolist()
                y_true_values[zip_code].append(y_test.tolist())
                y_predicted_values[zip_code].append(y_hat)

            for zip_code in list_of_zips:
                if zip_code in y_true_values and zip_code in y_predicted_values:
                    y_true_values[zip_code] = sum(y_true_values[zip_code],[])
                    y_predicted_values[zip_code] = sum(y_predicted_values[zip_code],[])


        for zip_code in list_of_zips:
            if zip_code in y_true_values and zip_code in y_predicted_values:
                mse_by_zip[zip_code]=mean_squared_error(y_true_values[zip_code],y_predicted_values[zip_code])

    return y_true_values, y_predicted_values, mse_by_zip
