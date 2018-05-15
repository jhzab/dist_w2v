#!/opt/anaconda3/bin/python3
from numpy import array
import sys
import os
import time
import logging
import pickle
from collections import defaultdict
from gensim.models import word2vec
from pyarrow import hdfs
from pyarrow.lib import ArrowIOError
from hdfs3.core import HDFileSystem

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('reducer')
logger.setLevel("INFO")

dimensions=500
PATH=f"/user/zab/w2v_wiki_{dimensions}_lr_300000"

logger.info("Initializing hdfs client")
#hdfs_client = hdfs.connect('master.ib')
hdfs_client = HDFileSystem('master.ib', port=8020)
words_seen = defaultdict()


def set_environ():
    pwd = os.environ["PWD"]
    os.environ.clear()
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/home/zab/bin:/usr/local/sbin:/usr/sbin"
    os.environ["PWD"] = pwd

def create_new_model():
    logger.info("Loading word dictionary")
    with open("wiki_dict_most_freq_300000","br") as dict_most_freq:
        word_freq = pickle.load(dict_most_freq)
        logger.info("Finished loaded word dictionary")
    model = word2vec.Word2Vec(size=dimensions, workers=9, iter=1, sg=1, window=10, compute_loss=True)
    logger.info("Building vocab from dictionary")
    model.build_vocab_from_freq(word_freq)
    logger.info("Finished building vocab")

    return model


def load_model(key, epoch):
    logger.info("Loading model: %s/%s" % (PATH, key))
    with hdfs_client.open("%s/%s_epoch_%d.model" % (PATH, key, epoch), 'rb') as model_fd:
        model = pickle.load(model_fd)
    logger.info("Model loaded")
    return model


def save_model(model, key, epoch):
    logger.info("Saving model")
    with hdfs_client.open("%s/%s_epoch_%d.model" % (PATH, key, epoch), 'wb') as model_fd:
        #model.save(model_fd, sep_limit=1024 * 1024, pickle_protocol=2)
        pickle.dump(model, model_fd, protocol=4)
    logger.info("Model saved :-)")

    
def convert_to_dict(model):
    for key in model.wv.vocab.keys():
        d = dict()
        d[key] = model.wv.get_vector(key).tolist()
        js = json.dumps(d)
        print(js)
        
    return d


def get_key():
    # this will have to read one line of input :(
    key, _ = next(sys.stdin).rstrip().split("\t")
    logger.info("This is model number: %s" % key)
    return key


def model_exists(key, epoch):
    return hdfs_client.exists("%s/%s_epoch_%d.model" % (PATH, key, epoch))

def get_current_epoch(key):
    for epoch in range(5, 0, -1):
        if model_exists(key, epoch):
            logger.info("Current epoch: %d" % (epoch + 1))
            return epoch + 1

    logger.info("Current epoch: 1")
    return 1


key = get_key()
current_epoch = get_current_epoch(key)

if current_epoch > 1:
    model = load_model(key, current_epoch - 1)
else:
    model = create_new_model()


def save_words_seen(key, epoch):
    logger.info("Saving words_seen")
    with hdfs_client.open("%s/words_seen_%s_epoch_%d.pkl" % (PATH, key, epoch), 'wb') as fd:
        pickle.dump(words_seen, fd, protocol=4)
    logger.info("Saved words_seen")


def process_input():
    seen = defaultdict(int)

    for line in sys.stdin:
        line = line.rstrip()
        k, value = line.split('\t')

        if key != k:
            logger.error("Keys differ. key: %s k: %s" % (key, k))
            logger.error("Line: %s" % line)
            exit(-1)

        tokens = value.split(' ')
        for token in tokens:
            seen[token] += 1
        yield tokens

    global words_seen
    words_seen = seen

sentences = process_input()

logger.info("Started training")
model.train(sentences, total_words=350000000, epochs=1)
logger.info("Finished training")

#convert_to_dict(model)

# pickling wont work because hadoop is adding \t characters at the end of a line for some reason4
#pmodel = pickle.dumps(model, protocol=4)
#sys.stdout.buffer.write(pmodel)

#set_environ()
save_model(model, key, current_epoch)
save_words_seen(key, current_epoch)
logger.info("Loss: %d" % model.get_latest_training_loss())
