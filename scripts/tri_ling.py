# imports
import pickle
import random
from gensim.models import Word2Vec

def main():
  languages = ['en','fr','jp'] 
  with open('../data/raw_data.pickle', 'rb') as x:
    data = pickle.load(x)
  alldata = []
  for lang in languages:
    alldata += data[lang]['corpus'].tolist()
  print('loaded')
  random.shuffle(alldata)
  model = Word2Vec(alldata,min_count=2,vector_size=200,workers=5,window=4,sg=1,negative=5) 
  print('training...') 
  model.train(alldata,total_examples=model.corpus_count,epochs=50)
  print('trained!') 
  with open('../embeddings/alltri.pickle','wb') as x:
    pickle.dump(model,x)
  print('dumped!')

if __name__ == '__main__':
  main()
  
   
    
