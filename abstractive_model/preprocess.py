import unicodedata
import string
import re
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

## Move models to GPU
USE_CUDA = True

########################################
## Parameters
########################################

## 训练数据所在位置，注意最后没有文件分隔符'/'
#FILE_PATH = './data/abstractive_input'
FILE_PATH = '/root/sharefolder/data/text_mining/sumdata/train' # gigawords in docker
SOS_token = 0
EOS_token = 1
## 句子最大长度，因为经过stage1，因此最大长度可以适当减少
#MAX_LENGTH = 150
MAX_LENGTH = 100

########################################
## Class To Index The Word
########################################

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3 # Count SOS and EOS and UNK
      
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
########################################
## Basic Preprocess Functions
########################################

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines1 = open(FILE_PATH+'/%s.txt' % lang1, encoding='utf8').read().strip().split('\n')
    lines2 = open(FILE_PATH+'/%s.txt' % lang2, encoding='utf8').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(lines1[x]), normalize_string(lines2[x])] for x in range(len(lines1))]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        voc = Voc(lang1+lang2)
    else:
        voc = Voc(lang1+lang2)
        
    return voc, pairs

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 
    # return True

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1_name, lang2_name, reverse=False):
    voc, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        voc.index_words(pair[0])
        voc.index_words(pair[1])

    return voc, pairs


########################################
## Basic Word Embedding Implementation
########################################

# Return a list of indexes, one for each word in the sentence
#def indexes_from_sentence(voc, sentence):
#    return [voc.word2index[word] for word in sentence.split(' ')]

def indexes_from_sentence(voc, sentence):
    sentence_index = []
    for word in sentence.split(' '):
        if word not in voc.word2index:
            ## index 2 mean 'UNK'
            sentence_index.append(2)
        else:
            sentence_index.append(voc.word2index[word])
    return sentence_index

def variable_from_sentence(voc, sentence):
    indexes = indexes_from_sentence(voc, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(voc, pair):
    input_variable = variable_from_sentence(voc, pair[0])
    target_variable = variable_from_sentence(voc, pair[1])
    return (input_variable, target_variable)