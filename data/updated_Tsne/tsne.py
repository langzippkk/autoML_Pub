

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

# Install the latest Tensorflow version.
!pip3 install --quiet "tensorflow>=1.7"
# Install TF-Hub.
!pip3 install --quiet tensorflow-hub
!pip3 install --quiet seaborn

oboe_embeddings = np.load('/content/new_oboe.npy')
auto_scikitlearn_embeddings =np.load('/content/new_autosklearn.npy')
kaggle = np.load('/content/new_kaggle.npy')

dataset_embeddings = oboe_embeddings[:,0:512]
kaggle = kaggle[:,0:512]
oboe_embeddings = oboe_embeddings[:,512:]
auto_scikitlearn_embeddings = auto_scikitlearn_embeddings[:,512:]

"""# ALL"""

all = np.concatenate((oboe_embeddings,auto_scikitlearn_embeddings),axis=1)
data =all
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x=X_embedded[:,0]
y=X_embedded[:,1]

n = range(1,45)
fig, ax = plt.subplots()
ax.scatter(x, y,s=200,color='y')

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
    
    
    
plt.axis('off')

"""# Oboe / AS"""

data = oboe_embeddings

## data = auto_scikitlearn_embeddings
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x=X_embedded[:,0]
y=X_embedded[:,1]

n = range(1,45)
fig, ax = plt.subplots()
ax.scatter(x, y,s=200,color='g')

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
    
    
plt.axis('off')

"""# Oboe,MD/ AS,MD"""

data = oboe_embeddings
## data = auto_scikitlearn_embeddings
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x=X_embedded[:,0]
y=X_embedded[:,1]

n =  range(1,45)
fig, ax = plt.subplots()
ax.scatter(x, y,s=200,color='r')

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
data = dataset_embeddings
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x1=X_embedded[:,0]
y1=X_embedded[:,1]   
    
n =  range(1,45)
ax.scatter(x1, y1,s=200,color='g')

for i, txt in enumerate(n):
    ax.annotate(txt, (x1[i], y1[i]))
    
plt.axis('off')    
ax.set_ylabel('Dimension 2')
ax.set_xlabel('Dimension 1')

"""# kaggle,Oboe,AS"""

data = kaggle
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x=X_embedded[:,0]
y=X_embedded[:,1]

n =  range(1,45)
fig, ax = plt.subplots()
ax.scatter(x, y,s=200,color='r')

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
data = oboe_embeddings
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x1=X_embedded[:,0]
y1=X_embedded[:,1]   
    
n =  range(1,45)
ax.scatter(x1, y1,s=200,color='g')

for i, txt in enumerate(n):
    ax.annotate(txt, (x1[i], y1[i]))
    
    
data = auto_scikitlearn_embeddings
X_embedded = TSNE(n_components=2,perplexity=15,random_state=1).fit_transform(data)
X_embedded.shape
x1=X_embedded[:,0]
y1=X_embedded[:,1]   
    
n =  range(1,45)
ax.scatter(x1, y1,s=200,color='y')

for i, txt in enumerate(n):
    ax.annotate(txt, (x1[i], y1[i]))   
    
    
plt.axis('off')    
ax.set_ylabel('Dimension 2')
ax.set_xlabel('Dimension 1')