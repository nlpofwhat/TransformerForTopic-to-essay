# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
from utils import calc_num_batches
import numpy as np
#*****************
# training_config["word_dict"]

def build_vocab(src_file_path,tag_file_path):
    char_dict =()
    char_dict['PAD'] = len(char_dict)
    char_dict['START'] = len(char_dict)
    char_dict['END'] = len(char_dict)
    with open(src_file_path,'r',encoding='utf-8') as f1:
        for line in f1:
            topics = list("".join(line.strip().split()))
            for c in topics:
                char_dict[c] = char_dict.get(c,0)+1
    with open(tag_file_path,'r',encoding='utf-8') as f2:
        for line in f2:
            topics = list("".join(line.strip().split()))
            for c in topics:
                char_dict[c] = char_dict.get(c,0)+1
    #保存字典 vocab_fpath

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    
    #vocab_dict = np.load(training_config["word_dict"]).item()
    token2idx = np.load(vocab_fpath).item()
    #print(token2idx)
    #这个npy是一个字典，word的列表的形式
    #vocab_dict = {w:id for id,w in enumerate(vocab_dict)}
    # 这个npy是一个字典，{word:id of word}的形式
    idx2token = {v: k for k, v in token2idx.items()}
    #print(idx2token)
    return token2idx, idx2token

#*****************
def load_data(fpath1, fpath2, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    
    with open(fpath1, 'r',encoding='utf-8') as f1, open(fpath2, 'r',encoding='utf-8') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            # 每个词按照空格分开，strip()去掉首尾的空格，得到字符串的列表
            sents2.append(sent2.strip())
    return sents1, sents2


# 由于原始数据已经进行了token2id的转化所以这个不需要
def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp_str = inp.decode("utf-8")
    #if type=="x": tokens = inp_str.split() + ["</s>"]
    if type=="x": tokens = inp_str.split() + ["<EOS>"]
    else: tokens = ["<GO>"] + inp_str.split() + ["<EOS>"]

    x = [dict.get(t, dict["<UNK>"]) for t in tokens]
    x = [dict.get(t, dict["<UNK>"]) for t in tokens]
    return x

#*****************
def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents  list of topic 
    sents2: list of target sents  list of essay 
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen,sent2)

def input_fn(sents1, sents2, vocab_fpath,batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents  topic path
    sents2: list of target sents  essay path
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None], (), ()), # shape of xs
              ([None], [None], (), ())) # shape of ys
    types = ((tf.int32, tf.int32, tf.string), # type of xs
             (tf.int32, tf.int32, tf.int32, tf.string)) # type of ys
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2,vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath1, fpath2, maxlen1, maxlen2,vocab_fpath,  batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.  topic path
    fpath2: target file path. string.  essay path
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2,vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
