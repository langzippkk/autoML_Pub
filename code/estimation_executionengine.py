import os
import json
import sys
import numpy as np
import pandas as pd
import time
import multiprocessing
from auto_learner import AutoLearner
import util

import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn import preprocessing as preproc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import estimator
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# disable warnings
import warnings

warnings.filterwarnings('ignore')

"""Different types of column data (need to differentiate in the Metadata):
* Numerical Features: Age, Fare, SibSp and Parch, \\
* Categorical Features: Sex, Embarked, Survived and Pclass, \\
* Alphanumeric Features: Ticket and Cabin(Contains both alphabets and the numeric value), \\
* Text Features: Name \\

**Pre-processing**: \\
Split test train data, \\
Treat missing values, \\
Normalization of columns with mean and variance, \\
Categorical to one hot encoding conversion, \\
Create polynomial features \\

"""

worksheet = pd.read_excel("final_autokaggle.xlsx", sheet_name='Metadata', index_col=0)


def preprocessing(train_df):
    X = train_df.drop(metadata['target_column'], 1)
    y = train_df[metadata['target_column']]
    existing_categorical = []  # existing columns (not necessarily in excel)
    existing_numerical = []  # existing columns (not necessarily in excel)

    if isinstance(y.values, (object, str)):  # make categorical
        le = LabelEncoder()
        y = le.fit_transform(y)
    X = X.filter(metadata['numeric_columns'] + metadata['categorical_columns'])
    # treat missing values
    pd.set_option('mode.chained_assignment', None)  # used to subside the panda's chain assignment warning
    X = X.replace('?', np.nan)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    for col in metadata['numeric_columns']:
        if col in X:
            X[[col]] = imp.fit_transform(X[[col]])
            existing_numerical.append(col)

    # Categorial transform (question1 and question2 split)
    # for col in metadata['categorical_columns']:
    #   if col in X:
    #       # existing_categorical.append(col)
    #       col_dummies = pd.get_dummies(X, columns=[col], dummy_na=True)
    #       X = col_dummies
    # Categorial transform (question1 and question2 not split)
    for col in metadata['categorical_columns']:
        if col in X:
            col_dummies = pd.get_dummies(X[col], dummy_na=True)
            X = pd.concat([X, col_dummies], axis=1)
            X.drop([col], axis=1, inplace=True)
            existing_categorical.append(col)

    # Feature normalization
    if existing_numerical:
        X[existing_numerical] = preproc.scale(X[existing_numerical])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15325)

    X_train = X_train.values
    X_test = X_test.values
    return X_train, X_test, y_train, y_test


def alpha_to_number(alpha_key):
    return sum([(ord(alpha) - 64) * (26 ** ind) for ind, alpha in enumerate(list(alpha_key)[::-1])]) - 1


def get_max_performance_metric(competition_name, row_idx):
    # returns the index of max performance metric
    subdf = worksheet.loc[worksheet['name'] == competition_name]
    # subdf = subdf.loc[subdf['estimator1 function call'].notnull()]
    perf = subdf.crossValidationPerformance.astype('float64')
    return perf.idxmax()


# Mapping from Metadata sheet column name to readable columns
column_key = {'name': 'C', 'columns': 'W', 'estimator_func_call': 'AU', 'target_name': 'AC', 'output_type': 'AA',
              'performance_metric': 'BB', 'feature_selector': 'AL'}
column_key = dict(map(lambda kv: (kv[0], alpha_to_number(kv[1])), column_key.items()))


def get_columns(row_id):
    return worksheet.loc[[row_id]]['columns'].values


def column_types(columns):
    columns = columns[0]
    columns_data = [x.strip() for x in columns[1:-1].split(';')]
    columns = []
    for ind, val in enumerate(columns_data):
        if ind % 3 == 2:
            if val in ['numeric', 'integer', 'real']:
                columns.append((columns_data[ind - 1], 'numeric'))
            elif val in ['categorical', 'boolean']:
                columns.append((columns_data[ind - 1], 'categorical'))
            elif val == 'string':
                columns.append((columns_data[ind - 1], 'text'))
            elif val == 'dateTime':
                columns.append((columns_data[ind - 1], 'dateTime'))
            else:
                pass
        elif ind % 3 == 1:
            if val == 'AgeuponOutcome':
                columns.append((columns_data[ind], 'numeric'))
    return columns


def parseMetaData(row_id):
    # Parse data from MetaData for each row
    metadata['competition_name'] = worksheet.loc[[row_id]]['name'].values
    metadata['task_type'] = worksheet.loc[[row_id]]['taskType'].values
    metadata['estimator'] = worksheet.loc[[row_id]]['estimator1 function call'].values
    metadata['estimator2'] = worksheet.loc[[row_id]]['estimator2'].values
    metadata['target_column'] = worksheet.loc[[row_id]]['targetName'].values
    metadata['output_type'] = worksheet.loc[[row_id]]['outputType'].str.split(',').values
    metadata['metric'] = worksheet.loc[[row_id]]['performanceMetric'].values
    metadata['feature_selector'] = worksheet.loc[[row_id]]['featureSelector'].values
    metadata['feature_selector'] = metadata['feature_selector'].astype(str)
    metadata['feature_selector'] = np.array(
        [x.lower() if isinstance(x, str) else x for x in metadata['feature_selector']])

    columns = get_columns(row_id)
    # Parse column information
    numeric_columns = [column.lower() for column, type_ in column_types(columns) if type_ == 'numeric']
    categorical_columns = [column.lower() for column, type_ in column_types(columns) if type_ == 'categorical']
    text_columns = [column.lower() for column, type_ in column_types(columns) if type_ == 'text']
    date_columns = [column.lower() for column, type_ in column_types(columns) if type_ == 'dateTime']
    metadata['numeric_columns'] = numeric_columns
    metadata['categorical_columns'] = categorical_columns
    metadata['text_columns'] = text_columns
    metadata['date_columns'] = date_columns
    # Remove target from features columns
    if metadata['target_column'] in metadata['numeric_columns']:
        metadata['numeric_columns'].remove(metadata['target_column'])
    if metadata['target_column'] in metadata['categorical_columns']:
        metadata['categorical_columns'].remove(metadata['target_column'])
    if metadata['target_column'] in metadata['text_columns']:
        metadata['text_columns'].remove(metadata['target_column'])
    if metadata['target_column'] in metadata['date_columns']:
        metadata['date_columns'].remove(metadata['target_column'])

    print(metadata['competition_name'])
    print(metadata['task_type'])
    print(metadata['numeric_columns'])
    print(metadata['text_columns'])
    print(metadata['date_columns'])
    print(metadata['categorical_columns'])
    print(metadata['target_column'])
    print("metadata['metric']: ", metadata['metric'])
    print("metadata['feature_selector']: ", metadata['feature_selector'])
    print("metadata['estimator']: ", metadata['estimator'])
    print("metadata['estimator2']: ", metadata['estimator2'])


def estimation(X_train, X_test, y_train, y_test, estimator_predicted):
    model = 0
    try:
        model = eval("".join(estimator_predicted))
        model.fit(X_train, y_train)
    except:
        if "sgdclassifier" in estimator_backup:
            estimator_predicted = "SGDClassifier(random_state=15325)"
            model = eval("".join(estimator_predicted))
            model.fit(X_train, y_train)

    print("estimator_predicted: ", estimator_predicted)

    if 'Sequential' in estimator_predicted or 'Dense' in "".join(estimator_predicted):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    prediction = model.predict(X_test)
    error_predicted = 0

    try:
        train_prediction = np.argmax(prediction, axis=1)
    except:
        train_prediction = prediction

    error_predicted = accuracy_score(y_test, train_prediction)
    print("accuracy using the predicted estimator: ", error_predicted)
    return float(error_predicted)


def Add_random_state(estimator_predicted):
    print("Before Add_random_state: ", estimator_predicted)
    if "RandomForestRegressor" in estimator_predicted:
        estimator_predicted = estimator_predicted.replace("RandomForestRegressor", "RandomForestClassifier")

    if "RandomForestClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            if "random_state=None" in estimator_predicted:
                estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
            elif "random_state= 0" in estimator_predicted:
                estimator_predicted = estimator_predicted.replace("random_state= 0", "random_state=15325")
            elif "random_state=0" in estimator_predicted:
                estimator_predicted = estimator_predicted.replace("random_state=0", "random_state=15325")
            elif "random_state=4141" in estimator_predicted:
                estimator_predicted = estimator_predicted.replace("random_state=4141", "random_state=15325")
            elif "random_state = 2016" in estimator_predicted:
                estimator_predicted = estimator_predicted.replace("random_state = 2016", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "MLPClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "SVC" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "LogisticRegression" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "ExtraTreesClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "XGBClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "AdaBoostClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")
    elif "DecisionTreeClassifier" in estimator_predicted:
        if "random_state" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace("random_state=None", "random_state=15325")
        elif "()" in estimator_predicted:
            estimator_predicted = estimator_predicted.replace(")", "random_state=15325 )")
        else:
            estimator_predicted = estimator_predicted.replace(")", ", random_state=15325 )")

    print("After Add_random_state: ", estimator_predicted)
    return estimator_predicted


"""# Running for 10 times and get the average"""

row_id_pairs = {169: 119}
#
n_epoch = 10
# apply the key's pipeline to the value's datasets


metadata = {}
V_value = 0

for i in range(n_epoch):
    # apply key's pipeline to value's dataset
    for key, value in row_id_pairs.items():

        # Parsing MetaData updates the metadata dict
        metadata.clear()
        print("************************************************************")
        parseMetaData(key)
        # get the predicted estimator
        estimator_predicted = metadata['estimator']

        if key == 113:
            estimator_predicted = "GradientBoostingClassifier(n_estimators=290, max_depth=9, subsample=0.5, learning_rate=0.01, min_samples_leaf=1, random_state=15325)"
        elif key == 144:
            estimator_predicted = "ExtraTreesClassifier(n_estimators=1200,criterion= 'entropy',min_samples_split= 2,max_depth= 30, min_samples_leaf= 2, n_jobs = -1, random_state=15325 )"
        elif key == 12:
            estimator_predicted = "RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features='auto', random_state=15325)"
        if isinstance(estimator_predicted, np.ndarray):
            estimator_predicted = estimator_predicted[0]
        elif isinstance(estimator_predicted, str):
            estimator_predicted = estimator_predicted
        estimator_backup = metadata['estimator2'][0]

        print("------------------------------------------------------------")

        metadata.clear()
        metadata = {}
        parseMetaData(value)
        train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/trainData.csv'
        if not os.path.exists("".join(train_csv_loc)):
            train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/train.csv'
        if os.path.exists("".join(train_csv_loc)):
            train_df = pd.read_csv(train_csv_loc[0])
        else:
            train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/train.json'
            with open(train_csv_loc[0]) as train_file:
                dict_train = json.load(train_file)
            train_df = pd.DataFrame(dict_train)
            train_df.reset_index(level=0, inplace=True)

        #       display(train_df)
        train_df.columns = map(str.lower, train_df.columns)
        train_df = train_df.applymap(lambda s: s.lower() if type(s) == str else s)

        X_train, X_test, y_train, y_test = preprocessing(train_df)
        if "MultinomialNB" in estimator_predicted:
            X_train_min = np.amin(X_train, axis=0)
            X_train = X_train - X_train_min

        if key == 7:
            input_shape = X_train.shape[1]
            print("X_train.shape: ", X_train.shape)
            estimator_predicted = """Sequential([Dense(4, activation="relu", kernel_initializer='random_normal', input_dim=""" + str(
                input_shape) + """), Dense(4, activation='relu', kernel_initializer='random_normal'), Dense(3, activation='sigmoid', kernel_initializer='random_normal')])"""

        # Add random_state
        if isinstance(X_train, np.ndarray) & isinstance(y_train, np.ndarray):
            estimator_predicted = Add_random_state(estimator_predicted)

            if key == 96:
                # create dataset for lightgbm
                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
                params = {'objective': 'binary', 'boosting': 'gbdt', 'learning_rate': 0.2, 'verbose': 0,
                          'num_leaves': 2 ** 8, 'bagging_fraction': 0.95, 'bagging_freq': 1, 'bagging_seed': 1,
                          'feature_fraction': 0.9, 'feature_fraction_seed': 1, 'max_bin': 256, 'num_rounds': 80,
                          'metric': 'auc'}
                gbm = lgb.train(params, lgb_train, 100, valid_sets=lgb_eval)
                y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                train_prediction = np.where(y_pred >= 0.5, 1, 0)
                e1 = accuracy_score(y_test, train_prediction)
                print("accuracy using the predicted estimator: ", e1)
            else:
                e1 = estimation(X_train, X_test, y_train, y_test, estimator_predicted)
                V_value = V_value + e1

        else:
            print("X_train or y_train is not type np.ndarray")

V_value = V_value / n_epoch
str_v = "Accuracy after running for "
str_v += str(n_epoch)
str_v += " times: " + str(V_value)
print(str_v)

print("************************************************************")

