import gensim
from gensim.models import Word2Vec 
import pickle
from collections import defaultdict
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

def dd2():
  return dict()

def dd():
  return defaultdict(dd2)

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

def main():
  languages = ['en','fr','jp']
  with open('../data/amazon_reviews/all_data.pickle', 'rb') as f:
    data = pickle.load(f)
  print('data loaded')

  for lang in languages:
    corpus = data[lang]['train']['corpus'][0:30000] + data[lang]['test']['corpus'][0:30000]
    model = Word2Vec(corpus,min_count=5, vector_size=200, workers=5, window=4,sg=1,negative=5,callbacks=[callback()],epochs=20,compute_loss=True)
    model.train(corpus,total_examples=model.corpus_count)
    print(lang, ' trained!')

    with open(f'../embeddings/mono/{lang}.pickle','wb') as x:
      pickle.dump(model,x)
    print(lang, ' dumped!')

if __name__ == '__main__':
  main()
