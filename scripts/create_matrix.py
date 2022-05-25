import pickle
import numpy as np
from collections import defaultdict
from procrustes import orthogonal

def get_emb_dict():
  emb_dict= defaultdict()
  monospaces = ['en','fr','jp']
  for lang in monospaces:
    emb_dict[lang] = defaultdict()
    emb_dict[lang]['matrix'],  emb_dict[lang]['lookup'] = get_matrix_lookup_mono(lang)
  
  emb_dict['enfr'] = defaultdict()
  emb_dict['enfr']['matrix'],  emb_dict['enfr']['lookup'] = get_matrix_lookup_cross(emb_dict['en'],emb_dict['fr'])
  emb_dict['enjp'] = defaultdict()
  emb_dict['enjp']['matrix'],  emb_dict['enjp']['lookup'] = get_matrix_lookup_cross(emb_dict['en'],emb_dict['jp'])

  return emb_dict

def get_matrix_lookup_mono(lang):
  with open(f'embeddings/mono/{lang}.pickle','rb') as f:
    w2v = pickle.load(f)
  word_mapping = w2v.wv.index_to_key # list of tokens
  
  matrix = w2v.wv.vectors   # (V,D)
  pad = np.zeros((1,200))
  matrix = np.vstack((pad,matrix))
  
  return matrix, word_mapping

def get_matrix_lookup_cross(emb_dict_1, emb_dict_2):
      
  l1_emb, l1_vocab = emb_dict_1['matrix'], emb_dict_1['lookup']
  l2_emb, l2_vocab = emb_dict_2['matrix'], emb_dict_2['lookup']

  # orthogonal Procrustes analysis with translation
  result = orthogonal(l1_emb, l2_emb, scale=False, translate=False, pad=True)

  # compute transformed matrix A (i.e., A x Q)
  AQ = np.dot(result.new_a, result.t) #np.allclose(AQ - new_B = True) 

  #find words that coexists in both vocabs
  coexisting_words = [word for word in l1_vocab if word in l2_vocab]

  for word in coexisting_words:
    l1_cont = l1_emb[l1_vocab.index(word),:]
    l2_cont = l2_emb[l2_vocab.index(word),:]
    shared_embeddings = np.add(l1_cont, l2_cont)*0.5
    # average of the two contributig vectors
    l1_emb[l1_vocab.index(word),:] = shared_embeddings # changing the embeddings in source language 
    l2_emb[l2_vocab.index(word),:] = shared_embeddings # l2_deletion.append(l2_vocab.index(word))

  bilingual_embedding_matrix = np.vstack((l1_emb, l2_emb))

  lookup = l1_vocab + l2_vocab

  return bilingual_embedding_matrix, lookup
