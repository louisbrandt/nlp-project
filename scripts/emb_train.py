import gensim
from gensim.models import Word2Vec 
import pickle
from collections import defaultdict
 
def dd2():
  return dict()

def dd():
  return defaultdict(dd2)

def main():
  languages = ['jp']
  with open('../data/all_data.pickle', 'rb') as f:
    data = pickle.load(f)
  print('data loaded')

  for lang in languages:
    corpus = data[lang]['train']['corpus'] + data[lang]['test']['corpus']
    model = Word2Vec(corpus,min_count=5, vector_size=200, workers=5, window=4, sg=1)
    model.train(corpus,total_examples=model.corpus_count,epochs=50)
    print(lang, ' trained!')

    with open(f'../embeddings/mono/{lang}.pickle','wb') as x:
      pickle.dump(model,x)
    print(lang, ' dumped!')

if __name__ == '__main__':
  main()
