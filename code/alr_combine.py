def gensim_as_dict(m):
    m.init_sims(replace=True)
    ret = {}
    for word in m.wv.vocab.keys():
        ret[word] = m.wv.get_vector(word)
    return ret

def gensim_as_dict_with_words_seen(pair):
    m, vocab = pair
    m.init_sims(replace=True)
    ret = {}
    for word, count in vocab.items():
        if word not in m.wv.vocab:
            continue
        ret[word] = m.wv.get_vector(word) 
    return ret

def load_wv_dict(mname):
    from gensim.models import Word2Vec
    m = Word2Vec.load(mname)
    m.init_sims()
    output = {}
    for word in m.wv.vocab:
        output[word] = m.wv.vectors_norm[m.wv.vocab[word].index]
    return output

def concat_on_common_indexes(vectors, indexes):
    v = [vecs[index] for vecs, index in zip(vectors, indexes)]
    v = np.hstack(v)
    return v

def indices(goodvalues, target_list):
    mask = np.isin(target_list, goodvalues)
    if mask.size == 1:
        return [True for i in range(len(target_list))], [i for i in range(len(target_list))]
    else:
        return mask, np.where(mask)[0]
    
def dim_reduce(origin_vecs, new_dim, use_norm):
    from sklearn.decomposition import TruncatedSVD
    trunc_svd = TruncatedSVD(n_components=new_dim, random_state=42)
    return trunc_svd.fit_transform(origin_vecs)

def eval_embedding(vocab_list, vector_matrix, dataset=None, cosine_similarity=False):
    w = Embedding.from_dict(dict(zip(vocab_list, vector_matrix)))
    if dataset is None:
        result = evaluate_on_all(w, cosine_similarity=cosine_similarity)
        return result
    if dataset == 'MEN':
        data = fetch_MEN()
        result = evaluate_similarity(w, data.X, data.y, cosine_similarity=False)
    elif dataset == 'Google':
        data = fetch_google_analogy()
        result = evaluate_analogy(w, data.X, data.y)
    elif dataset == 'RW':
        data= fetch_RW()
        result = evaluate_similarity(w, data.X, data.y, cosine_similarity=False)
    else:
        result = None
    return result

def randomly_drop_vectors(d, drop_words, k):
    vocab = list(d.keys())
    samples = random.sample(drop_words, k)
    for word in samples:
        del d[word]
    return d


from time import strftime, localtime
import pickle
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': program begins')
from functools import reduce
import numpy as np
from gensim.models import Word2Vec
import scipy
from copy import deepcopy
import logging
import random
import os
import sys

import pandas as pd

module_path = os.path.abspath(os.path.join('./src')) #  working around for import from relative dir in python3
if module_path not in sys.path:
    sys.path.append(module_path)

from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_on_all
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding #  working around end

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#========change this part to adapt experiment requirements==========================================
init_pca = True
evaluate = True
dump = True
base_folder = '/home/zijian/from_jan/w2v_wiki_500_lr_300000_sample_0.33/'
num_subs = 3
w2v_epoch = 5
subs_folder = base_folder + 'subs/'
words_folder = base_folder + 'words/'
res_m_folder = base_folder + 'result_models/'
output_folder = base_folder + 'output/'
logs_folder = base_folder + 'logs/'
#===================================================================================================


INFINITY = 10000000000000.
ERR_THRESHOLD = 5.
MAX_ITER = 3
CONCAT = -1
INIT = 0
dim_out = 500
dim = 500

os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())

flog = open(logs_folder + 'alr_logs.txt', 'a+')

print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to load sub-models')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to load sub-models\n')
models = [subs_folder + '{}_epoch_{}.model'.format(i, w2v_epoch) for i in range(num_subs)]
models = [Word2Vec.load(fname) for fname in models]
words = [words_folder + 'words_seen_{}_epoch_{}.pkl'.format(i, w2v_epoch) for i in range(num_subs)]
words = [pickle.load(open(fname, 'rb')) for fname in words]
ms_dict = list(map(gensim_as_dict_with_words_seen, zip(models, words)))
del models, words
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [end] sub-models loaded')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [end] sub-models loaded\n')

print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to align embeddings')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [begin] begin to align embeddings\n')
vocab_sets = list(map(lambda x: set(x.keys()), ms_dict))
vocab_union = reduce(lambda x, y: x.union(y), vocab_sets)
vocab_union = sorted(list(vocab_union))
vocab_common = reduce(lambda x, y: x.intersection(y), vocab_sets)
vocab_common = sorted(list(vocab_common))
index_common_in_union = indices(vocab_common, vocab_union)[1]
indexes_sub_in_union = []
count_hit = np.zeros(len(vocab_union))
indexes_common_in_sub = []
vectors = []
for md in ms_dict:
    vocab_sub = set(md.keys())
    vocab_sub = sorted(list(vocab_sub))
    count_hit_temp, index_sub_in_union = indices(vocab_sub, vocab_union)
    index_common_in_sub = indices(vocab_common, vocab_sub)[1]
    vecs = np.asarray([md[word] for word in vocab_sub])
    indexes_sub_in_union.append(index_sub_in_union)
    count_hit += count_hit_temp
    indexes_common_in_sub.append(index_common_in_sub)
    vectors.append(vecs)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [end] embeddings aligned')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [end] embeddings aligned\n')

print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to concatenate vectors')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [begin] begin to concatenate vectors\n')
vectors_concat = concat_on_common_indexes(vectors, indexes_common_in_sub)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [end] vectors concatenated')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [end] vectors concatenated\n')

if evaluate:
    epoch = CONCAT 
    fro_err = None
    o = pd.DataFrame(columns=["epoch", 'fronobius_error'], data = [[epoch, fro_err]])
    eval_res = eval_embedding(vocab_common, vectors_concat, cosine_similarity=True)
    output = pd.concat([o, eval_res], axis=1, join='inner')
    output.to_csv(output_folder + 'alr.csv')

print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to initialize, init_pca = {}'.format(init_pca))
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [begin] begin to initialize, init_pca = {}\n'.format(init_pca))
y = np.random.randn(len(vocab_union), dim)
if init_pca:
    y[index_common_in_union] = dim_reduce(vectors_concat, dim_out, False)
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [end] vectors for vocab union initialized')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [end] vectors for vocab union initialized\n')

if evaluate:
    epoch = INIT
    fro_err = INFINITY
    o = pd.DataFrame(columns=["epoch", 'fronobius_error'], data = [[epoch, fro_err]])
    eval_res = eval_embedding(vocab_common, y[index_common_in_union])
    output = output.append(pd.concat([o, eval_res], axis=1, join='inner'))
    output.to_csv(output_folder + 'alr.csv')

if dump:
    res_m_fname = res_m_folder+'epoch_'+str(epoch)+'.pkl'
    pickle.dump(dict(zip(vocab_union, y)), open(res_m_fname, 'wb+'))
    
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [begin] begin to train...')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [begin] begin to train...\n')

while epoch < MAX_ITER:
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()) +
          ': [begin] begin epoch %d'%(epoch+1))
    flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
               ': [begin] begin epoch %d\n'%(epoch+1))
    res = np.zeros_like(y)
    for vecs, index_sub_in_union in zip(vectors, indexes_sub_in_union):
        w, _ = scipy.linalg.orthogonal_procrustes(vecs, y[index_sub_in_union])
        temp = np.zeros_like(y)
        temp[index_sub_in_union] = np.dot(vecs, w)
        res += temp
    res/= count_hit.reshape(len(count_hit), 1)
    epoch += 1
    fro_err = np.linalg.norm(res - y, ord='fro')
    y = deepcopy(res)
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
          ': [end] epoch %d finished'%(epoch))
    flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
               ': [end] epoch %d finished\n'%(epoch))
    
    if evaluate:
        o = pd.DataFrame(columns=["epoch", 'fronobius_error'], data = [[epoch, fro_err]])
        eval_res = eval_embedding(vocab_union, y)
        output = output.append(pd.concat([o, eval_res], axis=1, join='inner'))
        output.to_csv(output_folder + 'alr.csv')
   
    if dump:
        res_m_fname = res_m_folder+'epoch_'+str(epoch)+'.pkl'
        pickle.dump(dict(zip(vocab_union, y)), open(res_m_fname, 'wb+'))
print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
      ': [end] all epochs of training and evaluations finished')
flog.write(strftime("%Y-%m-%d %H:%M:%S", localtime()) + 
           ': [end] all epochs of training and evaluations finished\n')

