# import some stuff
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scripts.lstm_train import LSTM, train, dd, dd2
from scripts.test import test, classify
import pickle
from sklearn.model_selection import train_test_split
from scripts.create_matrix import get_emb_dict


def pad(data, seq_len=200):
  padded = np.zeros((len(data), seq_len),dtype=int)
  for ii, review in enumerate(data):
    if len(review) != 0:
      padded[ii, -len(review):] = np.array(review)[:seq_len]
  return padded

def numerise(raw,lookup):
  data = []
  for review in raw: #[list of tokens]
    num_review = []
    for token in review:
      if token in lookup:
        num_review.append(lookup.index(token) + 1) # +1 for padding
      else:
        num_review.append(0)
    data.append(num_review)
  padded = pad(data)
  return padded



def main():
# get all prerequisites
  with open('data/amazon_reviews/raw_data.pickle','rb') as f:
    raw_data = pickle.load(f)
  big_embedding_dict = get_emb_dict()
  print('data & embeddings loaded!!')

# set languages
  lang1 = 'en'
  lang2 = 'en'

# load embeddings
  embeddings = big_embedding_dict['en']
  emb_matrix = torch.tensor(embeddings['matrix'])
  emb_lookup = embeddings['lookup']

  
# load raw training/testing data
  raw_lang1 = raw_data[lang1]['corpus'] 
  y_lang1 = raw_data[lang1]['y'] 
  
  raw_lang2 = raw_data[lang2]['corpus'] 
  y_lang2 = raw_data[lang2]['y'] 

  # pad and numerise reviews  
  if lang1==lang2:
    X = numerise(raw_lang1,emb_lookup) 
    y = y_lang1

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,shuffle=True,random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,random_state=69)
  else:
    X_train = numerise(raw_lang1,emb_lookup)
    y_train = y_lang1
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,random_state=69)
    
    X_test = numerise(raw_lang2,emb_lookup)
    y_test = y_lang2
    X_test, cutX, y_test, cuty = train_test_split(X_test, y_test, test_size=0.25,random_state=69)


  X_train = torch.tensor(X_train)
  X_val   = torch.tensor(X_val)
  X_test  = torch.tensor(X_test)
  y_train = torch.tensor(y_train)
  y_val   = torch.tensor(y_val)
  y_test  = torch.tensor(y_test)

  print("training language: {}".format(lang1)) 
  print("test language: {}".format(lang2)) 
  print("X_train size: {}".format(X_train.size()))
  print("X_val size: {}".format(X_val.size()))
  print("X_test size: {}".format(X_test.size()))
  print("y_train size: {}".format(y_train.size()))
  print("y_val size: {}".format(y_val.size()))
  print("y_test size: {}".format(y_test.size()))

  train_data = TensorDataset(X_train, y_train)  
  val_data = TensorDataset(X_val, y_val)  
  test_data = TensorDataset(X_test, y_test)  

# hyperparamers
  param_dict ={
      'hidden':400,
      'output':1,
      'emb':200,
      'layers':1,
      'dropout':0.8,
      'batch':64,
      'epochs':50,
      'lr':0.01
  }

# break data into batches
  batch_size = param_dict['batch'] 
  train_loader = DataLoader(train_data,batch_size=batch_size,drop_last=True)
  val_loader = DataLoader(val_data,batch_size=batch_size,drop_last=True)
  test_loader = DataLoader(test_data,batch_size=batch_size,drop_last=True)

# define the model
  model = LSTM(param_dict,emb_matrix)
  print('model defined!')

# run model on GPUs
  is_cuda = torch.cuda.is_available()
  if is_cuda:
      device = torch.device('cuda')
      print("GPU is available")
  else:
      device = torch.device('cpu')
      print("GPU not available, CPU used")
  model.to(device)

# run training loop
  print('initiating training...')
  model = train(model,param_dict,train_loader,val_loader,device)
  print('training completed')
  torch.save(model,'models/lstmenen.pt')
  print('model saved')

# test + evaluate
  print('testing in progress...')
  cf_matrix, f1 = test(model, test_loader,param_dict,device)
  print('testing completed')
  print('f1: ',f1) 
  print(cf_matrix)
 
if __name__ == '__main__':
  main()
