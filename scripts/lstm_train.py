# import some stuff
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
def dd2():
  return dict()

def dd():
  return defaultdict(dd2)

# define lstm class
class LSTM(nn.Module):

    def __init__(self, hyperparams, embeddings):
        super(LSTM,self).__init__() 
        
        # define params  
        self.hidden_dim = hyperparams['hidden']
        self.output_dim = hyperparams['output']
        self.embedding_dim = hyperparams['emb']
        self.num_layers = hyperparams['layers']
        self.batch_size = hyperparams['batch']

        # load pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(embeddings)

        # define lstm
        self.lstm = nn.LSTM(input_size = self.embedding_dim,
                            hidden_size = self.hidden_dim,
                            num_layers = self.num_layers,
                            bidirectional = True,
                            batch_first = True)
        
        # dropout layer
        self.dropout = nn.Dropout(hyperparams['dropout'])

        # fully connected component
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        
        # get embeddings for the input
        x = self.embedding(x) # (batch_size, seq_len, emb_dim)

        # lstm inference
        lstm_out, hidden = self.lstm(x.float(),hidden)
        
        # flatten lstm_out to 2D
        lstm_flat = lstm_out.contiguous().view(-1,self.hidden_dim)

        # dropout layer
        lstm_do = self.dropout(lstm_flat)

        # pass to fc
        fc_out = self.fc(lstm_do) 

        # sigmoid activation
        sigm = self.sig(fc_out)

        # format the output to prediciton vector
        out = sigm.view(self.batch_size, -1) 
        p = out[:,-1] 

        return p, hidden


def train(model,hyperparams,train_loader,valid_loader,device):

  epochs = hyperparams['epochs']
  lr = hyperparams['lr']
  batch_size = hyperparams['batch']
  hidden_dim = hyperparams['hidden']
  num_layers = hyperparams['layers']
  criterion = nn.BCELoss() 
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  losses = [] 
  min_valid_loss = np.inf
  training_losses = []
  validation_losses = []
  for epoch in range(epochs):
      # initialise lstm hidden states h = (h0,c0)
      model.train()
      h = (torch.zeros(num_layers*2,batch_size,hidden_dim).to(device),
           torch.zeros(num_layers*2,batch_size,hidden_dim).to(device)) 
      train_loss = 0.0
      for batch, y in train_loader:
          batch = batch.to(torch.int64)
          h = tuple([each.data for each in h]) # otherwise back prop doesn't work

          optimizer.zero_grad()

          # forward propegation
          p, h = model.forward(batch.to(device),h)

          # compute loss
          loss = criterion(p.squeeze(),y.float().to(device))
          train_loss += loss.item()

          # calculate gradients & update weights
          loss.backward()
          optimizer.step()

      valid_loss = 0.0
      model.eval()
      h = (torch.zeros(num_layers*2,batch_size,hidden_dim).to(device), torch.zeros(num_layers*2,batch_size,hidden_dim).to(device)) 
      for batch, y in valid_loader:
          batch = batch.to(torch.int64)
          p,h = model.forward(batch.to(device),h)

          loss = criterion(p.squeeze(),y.float().to(device))

          valid_loss += loss.item() 
      training_losses.append(train_loss/len(train_loader))     
      validation_losses.append(valid_loss/len(valid_loader))     
      print(f'Epoch {epoch+1} \t\t Training Loss: {training_losses[-1]} \t\t Validation Loss: {validation_losses[-1]}')
      if min_valid_loss > valid_loss:
          print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
          min_valid_loss = valid_loss
      
  return model, training_losses, validation_losses


