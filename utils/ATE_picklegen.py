import numpy as np
import pickle
from tqdm import tqdm
import torch
from textattack.shared import utils

def nearest_neighbours(emb_matrix, index, topn):
  embedding = torch.tensor(emb_matrix).to(utils.device)
  vector = torch.tensor(emb_matrix[index]).to(utils.device)
  dist = torch.norm(embedding - vector, dim=1, p=None)
  # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
  return dist.topk(topn + 1, largest=False)[1:].tolist()

max_neigh = 20
embedding_matrix = np.zeros((2000000, 300))
word2index = {}
index2word = {}
nn_matrix = np.zeros((2000000, max_neigh), dtype=np.uint32)
with open('cc.it.300.vec', mode='r') as input_file:
  for i, line in tqdm(enumerate(input_file)):
    if i > 0:
      word = line.split()[0]
      vec = line.split()[1:]
      embedding_matrix[i-1] = vec
      word2index[word] = i-1
      index2word[i-1] = word

print(f'START dump FILES\n')
with open('embedding_matrix.pkl', 'wb') as output:
  pickle.dump(embedding_matrix, output, protocol=4)
with open('word2index.pkl', 'wb') as output:
  pickle.dump(word2index, output, protocol=4)
with open('index2word.pkl', 'wb') as output:
  pickle.dump(index2word, output, protocol=4)

print(f'START NN MATRIX\n')
for i in tqdm(range(len(nn_matrix))):
  nn_matrix[i][:] = nearest_neighbours(embedding_matrix, i, max_neigh)

with open('nn.pkl', 'wb') as output:
  pickle.dump(nn, output, protocol=4)
