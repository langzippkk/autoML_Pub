import sys 
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer

import os
import json
import numpy as np
import pandas as pd
import time
import multiprocessing
import csv

import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn import preprocessing as preproc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score


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

tpot_results = {'index_kaggle': [], 'competition_name': [], 'pipeline': [], 'accuracy': []}
ranktable = pd.read_excel("RankTable.xlsx", sheet_name = 'index', index_col = 0)
index_pairing = {}
index_competition = {}
index_final = []
for row_id in range(len(ranktable)):
  index_pairing[row_id] = ranktable['corresponding_index'][row_id]
  index_competition[ranktable['competition_name'][row_id]] = row_id
index_final = ranktable['corresponding_index']


# tpot_results = {'index_kaggle': [], 'competition_name': [], 'pipeline': [], 'accuracy': []}
def save_to_excel(index_kaggle, competition_name, pipeline, accuracy):
  tpot_results['index_kaggle'].append(index_kaggle)
  tpot_results['competition_name'].append(competition_name)
  tpot_results['pipeline'].append(pipeline)
  tpot_results['accuracy'].append(accuracy)



"""## Helper functions"""

worksheet = pd.read_excel("final_autokaggle.xlsx", sheet_name = 'Metadata', index_col = 0)


def preprocessing(train_df):
#   print("metadata['target_column']: ", metadata['target_column'])
  X = train_df.drop(metadata['target_column'], 1)
  y = train_df[metadata['target_column']]
  existing_categorical = [] # existing columns (not necessarily in excel)
  existing_numerical = [] # existing columns (not necessarily in excel)

  
  if isinstance(y.values, (object, str)): # make categorical
    le = LabelEncoder()
    y = le.fit_transform(y)
  
  X = X.filter(metadata['numeric_columns'] + metadata['categorical_columns'])
  # treat missing values
  pd.set_option('mode.chained_assignment', None) # used to subside the panda's chain assignment warning
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
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  X_train = X_train.values
  X_test = X_test.values
  
  return X_train, X_test, y_train, y_test


def alpha_to_number(alpha_key):
  return sum([(ord(alpha)-64)*(26**ind) for ind, alpha in enumerate(list(alpha_key)[::-1])]) - 1


def get_max_performance_metric(competition_name, row_idx):
  # returns the index of max performance metric
  subdf = worksheet.loc[worksheet['name'] == competition_name]
  #subdf = subdf.loc[subdf['estimator1 function call'].notnull()]
  perf = subdf.crossValidationPerformance.astype('float64')
  return perf.idxmax()

def return_estimator(competition_name, task_type,performance_metric):
  # returns the index of max performance metric, filter first
  subdf = worksheet.loc[worksheet['name'] == competition_name]
  subdf = subdf.loc[subdf['estimators'].notnull()]
  subdf = subdf.loc[subdf['taskType'] == task_type]
  subdf = subdf.loc[subdf['performanceMetric']==performance_metric]
  perf = subdf.crossValidationPerformance.astype('float64')
  subdf = subdf.loc[perf.idxmax()]
  print('cross_validation_metric: ', subdf['crossValidationPerformance'])
  return subdf["estimators"]

# Mapping from Metadata sheet column name to readable columns
column_key = {'name': 'C', 'columns': 'W', 'estimator_func_call': 'AU', 'target_name': 'AC', 'output_type': 'AA', 'performance_metric': 'BB', 'feature_selector': 'AL'}
column_key = dict(map(lambda kv: (kv[0], alpha_to_number(kv[1])), column_key.items()))

def get_columns(row_id):
    return worksheet.loc[[row_id]]['columns'].values

def column_types(columns):
  columns = columns[0]
  columns_data = [x.strip() for x in columns[1:-1].split(';')]
  columns = []
  for ind, val in enumerate(columns_data):
    if ind%3 == 2:
      if val in ['numeric', 'integer', 'real']:
        columns.append((columns_data[ind-1], 'numeric'))
      elif val in ['categorical', 'boolean']:
        columns.append((columns_data[ind-1], 'categorical'))
      elif val == 'string':
        columns.append((columns_data[ind-1], 'text'))
      elif val == 'dateTime':
        columns.append((columns_data[ind-1], 'dateTime'))
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
  metadata['target_column'] = worksheet.loc[[row_id]]['targetName'].values
  metadata['output_type'] = worksheet.loc[[row_id]]['outputType'].str.split(',').values
  metadata['metric'] = worksheet.loc[[row_id]]['performanceMetric'].values
  metadata['feature_selector'] = worksheet.loc[[row_id]]['featureSelector'].values
  metadata['feature_selector'] = metadata['feature_selector'].astype(str)
  metadata['feature_selector'] = np.array([x.lower() if isinstance(x, str) else x for x in metadata['feature_selector']])

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
  print(metadata['metric'])
  print(metadata['feature_selector'])
  print(metadata['estimator'])
  
# Make a custom metric function
def my_custom_accuracy(y_true, y_pred):
  engine_accuracy = accuracy_score(y_true, y_pred)
  return engine_accuracy

# Make a custom a scorer from the custom metric function
# Note: greater_is_better=False in make_scorer below would mean that the scoring function should be minimized.
my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)

"""Obtain TPOT output pipeline and accuracy"""

metadata = {}
fromnow = False
return_estimators = []



for row_id in index_final:
    try:
      # Parsing MetaData updates the metadata dict
      metadata.clear()
      print("************************************************************")
      parseMetaData(row_id)
      train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/trainData.csv'
      if not os.path.exists("".join(train_csv_loc)):
        train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/train.csv'
      if os.path.exists("".join(train_csv_loc)):
        if metadata['task_type'][0] == 'classification':
          train_df = pd.read_csv(train_csv_loc[0])
          train_df.columns = map(str.lower, train_df.columns)
          train_df = train_df.applymap(lambda s:s.lower() if type(s) == str else s)
          X_train, X_test, y_train, y_test = preprocessing(train_df)            
        if isinstance(X_train, np.ndarray) & isinstance(y_train, np.ndarray):
          
          tpot = TPOTClassifier(population_size=40, verbosity=2, max_time_mins=3, max_eval_time_mins=2, scoring=my_custom_scorer)
          tpot.fit(X_train, y_train)
          print(tpot.score(X_test, y_test))
          
        else:
          print("X_train or y_train is not type np.ndarray")       
      print("************************************************************")
    except:
      print(sys.exc_info())

final_excel = pd.DataFrame.from_dict(tpot_results)



