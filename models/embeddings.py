import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import json
import requests
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

"""# Set up dataset and transform the data

**set up the dictionary for descriptions and the descrption of algorithm**
"""

import xlrd 
def set_dictionary(loc,number):

  wb = xlrd.open_workbook(loc) 
  sheet = wb.sheet_by_index(0) 
  new_dict = {}
  
  for i in range(1,number):
     word = sheet.cell_value(i,0)
  ## get the description of models  
     new_dict[word] = sheet.cell_value(i,2)
  return(new_dict)


def finddes(key):
 ## pass in a string

  sklearn = {
      # SVC
		"KNN" : "Classifier implementing the k-nearest neighbors vote. KNN is a non-parametric lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point",
		"DT" : "Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.",
		"RF" : "A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).",
		"GBT" : "Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.",
		"AB" : "An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.",
		"lSVM" : "More formally, a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.[3] Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier.[4]",
    "kSVM" : "More formally, a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.[3] Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier.[4]",
     "Logit": "In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one.",
      "Perceptron":"In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.[1] It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.",
      "GNB":"naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.",
      "MLP":"A multilayer perceptron (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.[1][2] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[3]",
      "ExtraTrees":"This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.",
      "Lasso":"Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters)",
      "Ridge":"Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value.",
      "ElasticNet":"Elastic net regularization. In statistics and, in particular, in the fitting of linear or logistic regression models, the elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.",
    	"SGD":"This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning, see the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.",
      "nb": "Naive Bayes classifier for multivariate Bernoulli models.",
      "DUMMY":"large large large dataset",
      "memory out":"large large large dataset",
      "CNN":" convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.CNNs are regularized versions of multilayer perceptrons. ",
      "NN":" Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. ",
      "GB":"A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.",
      "LDA":"A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.",  
      "passive_aggressive":"Passive Aggressive Algorithms are a family of online learning algorithms (for both classification and regression) proposed by Crammer at al.",
       "QDA":"A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes rule."
  
  }

  return sklearn.get(key)
  


def set_dictionary_pipeline(loc,number):

  wb = xlrd.open_workbook(loc) 
  sheet = wb.sheet_by_index(0) 
  new_dict1 = {}
  
  for i in range(1,number):
    temp=[]
    name = sheet.cell_value(i,0)
    key = sheet.cell_value(i,1)
    for word in key.split(","):
  ## get the description of models  
      temp.append(finddes(word))
    makeitastring = ''.join(map(str, temp))
    new_dict1[name] = makeitastring

  ## get the description of models  
  return(new_dict1)
  


"""**get embedding functions for certain input and competition names**"""

def get_embeddings1(dictionary,name):
  messages = [ dictionary [name] ]

  # Reduce logging output.
  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed(messages))

  return(message_embeddings)

# get_embeddings1(pipeline, 'kobe-bryant-shot-selection')

"""# Embeddings for Kaggle and final result"""

def test_embedding(doc,number):
  description=set_dictionary(doc,number)
  pipeline = set_dictionary_pipeline(doc,number)

  embeddings4 = np.zeros(shape=(1,512))
  embeddings5 = np.zeros(shape=(1,512))
  
  
## open the test case
  wb = xlrd.open_workbook(doc) 
  sheet = wb.sheet_by_index(0) 
  
## description embedding
  for i in range(1,number):
    print(i)
    embeddings4 = np.vstack((embeddings4,  get_embeddings1(description,  sheet.cell_value(i,0))     ) )
    
  embeddings4 = np.delete(embeddings4, 0, 0)

## oboe embedding
  for i in range(1,number):
      print(i)
      embeddings5 = np.vstack((embeddings5,  get_embeddings1(pipeline,  sheet.cell_value(i,0))     ) )
    
  embeddings5 = np.delete(embeddings5, 0, 0)
  
  oboe_and_meta = (np.concatenate((embeddings4,embeddings5), axis=1)) 
  print(len(oboe_and_meta))
  return(oboe_and_meta)


"""1. generate embeddings for kaggle all 44
2. build the target matrix (512,30),and pariwise distances
3. build the NN
4. put the test case into NN , get the distances. should be 12 *30

5. pick the smallest distances for each competition(the best solution)
6. compare it to the distance of real closest neighbor.

## generate embedding for TPOT
"""

tpot_embedding = test_embedding("TPOT_Metadata.xlsx",45)

# save output of embedding
np.save('new_tpot.npy', tpot_embedding)

"""## generate embedding for kaggle"""


kaggle1 = test_embedding("kaggle.xlsx",45)

# save output of embedding
np.save('new_kaggle.npy', kaggle1)

"""## calculate Euclidean distance matrix of kaggle data"""

## 2.
import numpy as np
kaggle1 = np.load('new_kaggle.npy')
def get_distance(data):
  n = len(data)
  d = np.zeros((n,n))

  for i in range(0,n):
  
    for j in range(0,n):
      temp =np.linalg.norm(data[i]-data[j])
      d[i][j] = temp
  return(d)

d = get_distance(kaggle1)

np.save('new_distance_matrix',d)
# Oboe_Metadata

"""## calculate KL matrix of kaggle data"""

def kl_divergence(p, q): 
    return tf.reduce_sum(p * tf.log(p/q)).eval(session=sess)
    
def get_kl_divergence(data):
  n = len(data)
  d = np.zeros((n,n))
  for i in range(0,n):
    data[i] = sess.run(tf.nn.softmax(data[i]))
  for i in range(0,n):
    for j in range(0,n):
      temp = kl_divergence(data[i], data[j])
      d[i][j] = temp
  return d

sess = tf.Session()
import numpy as np
kaggle1 = np.load('new_kaggle.npy')
data = kaggle1
display(data)
n = len(data)

d = get_kl_divergence(data)
np.save('KL_distance_matrix',d)

"""## generate embedding for Oboe"""

oboe_embedding = test_embedding("Oboe_Metadata.xlsx",45)

# save output of embedding
np.save('new_oboe.npy', oboe_embedding)

"""## generate embedding for Autosklearn"""

autosklearn_embedding = test_embedding("autosklearn_Metadata.xlsx",45)

# save output of embedding
np.save('new_autosklearn.npy', autosklearn_embedding)

"""3 and 4 are in there individual ipynb file




"""## Prediction based on Euclidean distance"""

ranktable = pd.read_excel("RankTable.xlsx", sheet_name = 'index', index_col = 0)
# display(ranktable)
index_pairing = {}
index_competition = {}
index_final = []
rank_to_kaggle = {}
for row_id in range(len(ranktable)):
  index_pairing[ranktable['corresponding_index'][row_id]] = row_id
  rank_to_kaggle[row_id] = ranktable['corresponding_index'][row_id]
  index_competition[row_id] = ranktable['competition_name'][row_id]
index_final = ranktable['corresponding_index']

oboe_embeddings =  np.load('new_oboe.npy')
auto_scikitlearn_embeddings = np.load('new_autosklearn.npy')
tpot_embeddings = np.load('new_tpot.npy')

oboe_md_embedding = oboe_embeddings
autosklearn_md_embedding = auto_scikitlearn_embeddings

dataset_embeddings = oboe_embeddings[:,0:512]
oboe_embeddings = oboe_embeddings[:,512:]
auto_scikitlearn_embeddings = auto_scikitlearn_embeddings[:,512:]
tpot_embeddings = tpot_embeddings[:,512:]





"""## Euclidean distance prediction"""

##5 & 6.

# d =  np.load('new_distance_matrix.npy')
# d =  np.load('KL_distance_matrix.npy')

# d = get_distance(oboe_embeddings)
# d = get_distance(auto_scikitlearn_embeddings)
# d = get_distance(tpot_embeddings)
# d = get_distance(dataset_embeddings)
# d = get_distance(oboe_md_embedding)
# d = get_distance(autosklearn_md_embedding)



result = np.zeros((44,44))
for j in range(0,len(d)):
  result[j]=np.argsort(d[j,])[:44]


test = []
for i in range(44):
  test.append(i)


final_result = []
correspond_index = []
for i in range(len(test)):
  orig = result[test[i]][0:].tolist()
  # remove the index of itself
  orig.remove(i)
  
  final_result.append(orig[0])
  correspond_index.append(rank_to_kaggle[int(orig[0])])
  # print out the index of the most nearest competition without its index
  print(result[test[i]][0:])
  print(orig)
  print("------------------------------------------------------------")

print("Test data in ranktable: ", test)  
# print("Test data corresponding to final_kaggle: ", test_index) 
  
print("Result in ranktable: ", final_result)
print("------------------------------------------------------------")
print("Result corresponding to final_kaggle: ", correspond_index)


## for each distance in increasing order, pick the lowest one in the train set

for i in range(len(correspond_index)):
  print(i, correspond_index[i])



