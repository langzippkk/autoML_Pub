
import os
import json
import numpy as np
import pandas as pd
import sys
import time
import multiprocessing

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

from sklearn.preprocessing import Imputer
from sklearn import preprocessing as preproc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import autosklearn.classification
import sklearn.model_selection
import warnings
import sklearn.datasets
import sklearn.metrics
from auto_learner import AutoLearner
import util
import csv
from ast import literal_eval

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


"""# Save output to excel"""
def save_to_excel(index_kaggle, competition_name, pipeline, accuracy):
  autosklearn_results['index_kaggle'].append(index_kaggle)
  autosklearn_results['competition_name'].append(competition_name)
  autosklearn_results['pipeline'].append(pipeline)
  autosklearn_results['accuracy'].append(accuracy)

worksheet = pd.read_excel("final_autokaggle.xlsx", sheet_name = 'Metadata', index_col = 0)

"""# Preprocessing the data"""

def preprocessing(train_df):
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
  imp = Imputer(missing_values=np.nan, strategy='mean')
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




"""Obtain Autosklearn output pipeline and accuracy"""
metadata = {}
seenset = set()
successfulrun = set()
fromnow = False
autosklearn_results = {'index_kaggle': [], 'competition_name': [], 'pipeline': [], 'accuracy': []}
ranktable = pd.read_excel("RankTable.xlsx", sheet_name = 'index', index_col = 0)
# display(ranktable)
index_pairing = {}
index_competition = {}
index_final = []
for row_id in range(len(ranktable)):
  index_pairing[row_id] = ranktable['corresponding_index'][row_id]
  index_competition[ranktable['competition_name'][row_id]] = row_id
index_final = ranktable['corresponding_index']
#####################
for row_id in index_final:
    try:
      # Parsing MetaData updates the metadata dict
      metadata.clear()
      print("************************************************************")
      parseMetaData(row_id)
      if metadata['competition_name'][0] in seenset:
        print('saw this one already')
      elif metadata['competition_name'][0] not in ['facebook-recruiting-iv-human-or-bot', 
                                                   'talkingdata-adtracking-fraud-detection', 'dogs-vs-cats-redux-kernels-edition',
                                                  'jigsaw-toxic-comment-classification-challenge', 'two-sigma-connect-rental-listing-inquiries',
                                                  'quora-question-pairs', 'spooky-author-identification', 'dogs-vs-cats','15-071x-the-analytics-edge-competition-spring-2015',
                                                  'uciml_sms-spam-collection-dataset', 'expedia-hotel-recommendations', 'word2vec-nlp-tutorial',
                                                  'austin-animal-center-shelter-outcomes-and', 'whats-cooking','march-machine-learning-mania-2017',
                                                  'intel-mobileodt-cervical-cancer-screening', 'landmark-recognition-challenge', 'imaterialist-challenge-furniture-2018',
                                                   'crowdflower-search-relevance', 'epa-air-quality', 'uciml-sms-spam-collection-dataset', 'zynicide/wine-reviews',
                                                                                                                           'two-sigma-financial-modeling',
                                                                                                                           'daliaresearch-basic-income-survey-european-dataset',
                                                                                                                           'uciml-faulty-steel-plates',
                                                                                                                           'ultrajack modern-renaissance-poetry',
                                                                                                                           'mrisdal/fake-news', 'uciml-forest-cover-type-dataset',
                                                                                                                           'yelp-dataset/yelp-dataset', 'slothkong/10-monkey-species']: # some skipped due to large size
        seenset.add(metadata['competition_name'][0])
        best_row_id = row_id
        if not best_row_id:
          print('no estimator function call or cross validation measure provided')
        else:
          train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/trainData.csv'
          if not os.path.exists("".join(train_csv_loc)):
            train_csv_loc = 'datasets/' + metadata['competition_name'] + '/data/train.csv'
          if os.path.exists("".join(train_csv_loc)):
            if metadata['task_type'][0] == 'classification':
              train_df = pd.read_csv(train_csv_loc[0])
#               display(train_df)
              train_df.columns = map(str.lower, train_df.columns)
              train_df = train_df.applymap(lambda s:s.lower() if type(s) == str else s)
              X_train, X_test, y_train, y_test = preprocessing(train_df)
              automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=50)
              automl.fit(X_train, y_train)
              y_hat = automl.predict(X_test)
              engine_accuracy = accuracy_score(y_test, y_hat)
              print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
              final_model = automl.get_models_with_weights()
              print("Models: ", final_model)
              auto_model = 0
              prob = 0
              for m in final_model:
                temp = m[0]
                if prob<temp:
                  auto_model = m[1]
              print("final pipeline: ", auto_model)
              
              # Save output to excel
              save_to_excel(index_competition[metadata['competition_name'][0]], metadata['competition_name'][0], auto_model, engine_accuracy)
              
              successfulrun.add(metadata['competition_name'][0])
            else:
              print('skipping for now: ', metadata['task_type'][0])
          else:
            print('no training data')
      print("************************************************************")
    except:
      print(sys.exc_info())
      if metadata['competition_name'][0] in seenset:
        seenset.remove(metadata['competition_name'][0])
      pass
print('seenset: ', seenset)
print(len(seenset))
print('successfulrun: ', successfulrun)
print(len(successfulrun))

final_excel = pd.DataFrame.from_dict(autosklearn_results)
