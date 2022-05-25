import numpy as np
import pickle 
import sklearn as sk
from collections import defaultdict

def dd2():
  return dict()
def dd():
  return defaultdict(dd2)


def main():
  with open('../data/all_data.pickle','rb') as x:
    all_data = pickle.load(x)
  languages =['en','fr','jp']
  new_data = defaultdict() 
  new_data['en'] = defaultdict()
  new_data['fr'] = defaultdict()
  new_data['jp'] = defaultdict()
  for lang in languages:
    X_train1 = all_data[lang]['train']['corpus']
    X_train2 = all_data[lang]['test']['corpus']
    y_train1 = all_data[lang]['train']['y']
    y_train2 = all_data[lang]['test']['y']
    X = np.concatenate((X_train1, X_train2))
    y = np.concatenate((y_train1, y_train2))
    new_data[lang]['corpus'] = X 
    new_data[lang]['y'] = y 
  with open('../data/raw_data.pickle','wb') as f:
    pickle.dump(new_data,f)
  print('done')

if __name__== '__main__':
  main()
