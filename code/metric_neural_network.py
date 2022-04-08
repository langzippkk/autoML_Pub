import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, LeakyReLU, concatenate, LSTM, Masking, TimeDistributed, Bidirectional, Activation, Embedding, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle



# load relevant files: kaggle.npy, oboe.npy, autosklearn.npy and distance_matrix.npy
oboe_embeddings = np.load('oboe.npy')
auto_scikitlearn_embeddings = np.load('autosklearn.npy')
tpot_embeddings = np.load('tpot.npy')

"""# Using symmetric distance"""
# distance_kaggle = np.load('new_distance_matrix.npy')

"""# Using asymmetric distance"""
distance_kaggle = np.load('KL_distance_matrix.npy')

# take the first 512 dimensions, readjust the embeddings
dataset_embeddings = oboe_embeddings[:,0:512]
oboe_embeddings = oboe_embeddings[:,512:]
auto_scikitlearn_embeddings = auto_scikitlearn_embeddings[:,512:]
tpot_embeddings = tpot_embeddings[:,512:]
print('datasets and their closest neighbors')
minimum_idx = np.argmin(distance_kaggle, axis=1)


ranktable = pd.read_excel("RankTable.xlsx", sheet_name = 'index', index_col = 0)
index_pairing = {}
index_competition = {}
index_final = []
rank_to_kaggle = {}
for row_id in range(len(ranktable)):
  index_pairing[ranktable['corresponding_index'][row_id]] = row_id
  rank_to_kaggle[row_id] = ranktable['corresponding_index'][row_id]
  index_competition[row_id] = ranktable['competition_name'][row_id]
index_final = ranktable['corresponding_index']


# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return (1 - SS_res/(SS_tot + backend.epsilon()))

def data_prep(dataset, labels):
      indicies = len(labels)
      indices_train = []
      for i in range(indicies):
        if i not in indices_test:
          indices_train.append(i)
      X_train = dataset[indices_train, :]
      X_test = dataset[indices_test, :]
      y_train = labels[indices_train, :]
      y_test = labels[indices_test, :]


      # exclude those distance which points back to themselves
      y_train = y_train[:, indices_train] # only keep the rows and columns in our "database"
      y_test = y_test[:, indices_train]
      return (X_train, X_test, y_train, y_test, np.asarray(indices_train), np.asarray(indices_test))




"""Run each test dataset"""

inputs_kaggle = []
closest_kaggle = []
closest_ranktable = []
ground_truth_rank = []

test_set = []
for i in range(44):
  test_set.append(i)


"""inputs and outputs"""


""" 1. dataset embeddings """
# dataset = dataset_embeddings

""" 2. oboe embeddings """
# dataset = oboe_embeddings

""" 3. autosklearn embeddings """
# dataset = auto_scikitlearn_embeddings

""" 4. tpot embeddings """
# dataset = tpot_embeddings

""" 5. concatenation of dataset and oboe embeddings """
# dataset = np.append(dataset_embeddings, oboe_embeddings, axis=1)

""" 6. concatenation of dataset and autosklearn embeddings """
# dataset = np.append(dataset_embeddings, auto_scikitlearn_embeddings, axis=1)

""" 7. concatenation of dataset and tpot embeddings """
# dataset = np.append(dataset_embeddings, tpot_embeddings, axis=1)

""" 8. all three of dataset, autosklearn and tpot embeddings """
dataset = np.append(dataset_embeddings, oboe_embeddings, axis=1)
dataset = np.append(dataset, auto_scikitlearn_embeddings, axis=1)
dataset = np.append(dataset, tpot_embeddings, axis=1)
  
for test_ind in test_set:
  indices_test = [test_ind]

  labels = distance_kaggle
  X_train, X_test, y_train, y_test, indices_train, indices_test = data_prep(dataset, labels)
  

  # parameters
  loss = 'mean_squared_error'
  early_stopping_loss = loss
  mid_activation='relu'
  final_activation='linear'
  batch_size = 16
  epochs = 1200
  optimizer = tf.optimizers.Adam(0.001)

  filepath = 'model.h5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  earlystopping = EarlyStopping(monitor=early_stopping_loss, patience=200, verbose=1, mode='auto')
  log_loc = "./logs"
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_loc, histogram_freq=1,
                           write_graph=True,
                           write_grads=True,
                           batch_size=batch_size,
                           write_images=True,
                           update_freq = 'epoch')
  callbacks_list = [checkpoint, earlystopping, tensorboard_callback]
  inputs = Input(shape=(X_train.shape[1],))
  x = (Dense(units=128, activation=mid_activation))(inputs)
  x = (Dense(units=64, activation=mid_activation))(x)
  x = (Dense(units=64, activation=mid_activation))(x)
  # final activation, learned distances to existing datasets
  x = (Dense(units=y_train.shape[0], activation=final_activation))(x)
  model = Model(inputs=inputs, outputs=x)
  model.summary()

  model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[loss, rmse, r_square]
    )
  result = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15, callbacks=callbacks_list)
  model.evaluate(X_test,y_test)
  # get predictions
  
  
  y_pred = model.predict(X_test)
  minimum_idx = np.argmin(y_pred, axis=1)
  print('test idx in ranktable: ', indices_test)
  print('closest predicted neighbors idx: ', indices_train[minimum_idx])
  closest_ranktable.append(indices_train[minimum_idx])

  final_re = []
  original_in = []
  for i in range(len(indices_test)):
    original_in.append(rank_to_kaggle[indices_test[i]])
    final_re.append(rank_to_kaggle[indices_train[minimum_idx][i]])
  print("----------------------------------------------------------")
  print("original inputs: ", original_in)
  inputs_kaggle.append(original_in)
  print("closest predicted neighbors in final_kaggle: ", final_re)
  closest_kaggle.append(final_re)
  print("----------------------------------------------------------")

  # Get the rank of the current pipeline in the ground truth rank table.
  rankarr=[]
  for didx in range(len(indices_test)):
      sorted_idx = np.argsort(distance_kaggle[indices_test[didx],])
      rank = list(sorted_idx).index(indices_train[minimum_idx][didx])
      print('curr idx: ', didx, 'ground truth rank: ', rank)
      rankarr.append(rank)
  ground_truth_rank.append(rank)

print("original inputs in kaggle: ", inputs_kaggle)
print("closest predicted neighbors in final_kaggle: ", closest_kaggle)
print("closest predicted neighbors in rank table: ", closest_ranktable)
print("rank of predicted pipeline in ground truth rank: ", ground_truth_rank)