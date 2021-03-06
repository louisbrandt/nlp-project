import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
import numpy as np

def test(model,test_loader,params,device):
  y_true =[] 
  y_pred =[] 
  batch_size = params['batch'] 
  hidden_dim = params['hidden']
  num_layers = params['layers'] 
  h = (torch.zeros(num_layers*2,batch_size,hidden_dim,).to(device),torch.zeros(num_layers*2,batch_size,hidden_dim,).to(device))
  for batch, y in test_loader:
    batch = batch.to(torch.int64)

    output,h = model(batch.to(device),h)   #[0.2,0.4,...,0.9,0.1] length = batch_size
    p = [classify(_.item()) for _ in output]

    y_pred.append(p)
    y_true.append(y.tolist()) 
  y_pred = np.array(y_pred).reshape(-1,1)
  y_true = np.array(y_true).reshape(-1,1)
  print(classification_report(y_true, y_pred, digits=3))
  accuracy = accuracy_score(y_true, y_pred)
  return confusion_matrix(y_true,y_pred), f1_score(y_true, y_pred, average='macro'), accuracy


def classify(p):
  if p > 0.5:
    return 1
  return 0 
