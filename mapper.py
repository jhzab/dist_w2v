#!/opt/anaconda3/bin/python3

import sys
import json
import string
import re
#import logging
import numpy as np
from collections import deque

#FORMAT = '%(asctime)-15s %(message)s'
#logging.basicConfig(format=FORMAT)
#logger = logging.getLogger('reducer')
#logger.setLevel("INFO")

SAMPLE = 0.10
NUMBER_REDUCERS = 10
max_queue_len = 100000
epochs = 1
CHOICES = list(range(0, NUMBER_REDUCERS))


re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
whitespace = re.compile(r"^\s+$")
whitespace_all = re.compile(r"\s+")
money = re.compile(r"^\d+([\.,]\d+)?[€$£₤]?\.?$")
number = re.compile(r"^\d+(.\d+)?\.?$")
alpha = re.compile(r"\w")


def filter_tokens(tokens):
    filter(lambda token: not(whitespace.match(token) or token == ""))


def remove_punctuation(text):
    return re_tok.sub(r' ', text)


def merge_whitespace(text):
    return whitespace_all.sub(r' ', text)


def remove_and_sanitize_tokens(tokenized_sentence):
    # need to check for numbers, money and http urls before removing special tokens.
    # otherwise the regexes will not match
    return list(
        filter(
            lambda token: not number.match(token) and not money.match(token) and not token.startswith("http:") and not token.startswith("https:"),
            map(lambda token: token.lower(), tokenized_sentence)))


def remove_and_sanitize_text(text):
    text = remove_punctuation(text)
    text = merge_whitespace(text)
    return " ".join(remove_and_sanitize_tokens(text.split()))

def get_keys():
    return list(filter(lambda key: True if np.random.uniform() <= SAMPLE else False, CHOICES))


def get_at_least_one_key():
    keys = get_keys()
    while len(keys) == 0:
        keys = get_keys()

    return keys


# the queue will save entries of type (key: int, pragraph: str, ctr: int)
# pararaph is just a paragraph that is fed into w2v and ctr gives
# the number of times the paragraph can still be used
queue = deque()


def print_from_queue():
    key, paragraph, ctr = queue.popleft()
    print("%d\t%s" % (key, paragraph))

    if ctr > 1:
        queue.append( (key, paragraph, ctr - 1) )


def send_from_queue():
    while len(queue) >= max_queue_len:
        print_from_queue()


def deplete_queue():
    try:
        while True:
            print_from_queue()
    except IndexError:
        return


def get_data_from_json():
    for line in sys.stdin:
        cnt = 0
        js = json.loads(line)

        paragraphs = js["par"]
        for p in paragraphs:
            p = p["text"]
            #send_from_queue()

            normalized = remove_and_sanitize_text(p)
            yield normalized


def get_data_from_stdin():
    for line in sys.stdin:
        yield line.rstrip()


if __name__ == '__main__':
    for line in get_data_from_json():
        for key in get_at_least_one_key():
            if epochs - 1 >= 1:
                queue.append( (key, line, epochs - 1) )
            print("%d\t%s" % (key, line))

    deplete_queue()
