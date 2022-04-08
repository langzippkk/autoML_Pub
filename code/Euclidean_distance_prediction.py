import xlrd 
import tensorflow as tf
import numpy as np
import pandas as pd


def get_distance(data):
  n = len(data)
  d = np.zeros((n,n))

  for i in range(0,n):
  
    for j in range(0,n):
      temp =np.linalg.norm(data[i]-data[j])
      d[i][j] = temp
  return(d)


"""## Prediction based on Euclidean distance"""

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

oboe_embeddings =  np.load('oboe.npy')
auto_scikitlearn_embeddings = np.load('autosklearn.npy')
tpot_embeddings = np.load('tpot.npy')

oboe_md_embedding = oboe_embeddings
autosklearn_md_embedding = auto_scikitlearn_embeddings

dataset_embeddings = oboe_embeddings[:,0:512]
oboe_embeddings = oboe_embeddings[:,512:]
auto_scikitlearn_embeddings = auto_scikitlearn_embeddings[:,512:]
tpot_embeddings = tpot_embeddings[:,512:]


"""## Euclidean distance prediction"""
# load the embeddings

# d =  np.load('new_distance_matrix.npy')
# d = get_distance(oboe_embeddings)
# d = get_distance(auto_scikitlearn_embeddings)
d = get_distance(tpot_embeddings)
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
  
print("Result in ranktable: ", final_result)
print("------------------------------------------------------------")
print("Result corresponding to final_kaggle: ", correspond_index)