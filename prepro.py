# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)

def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    # 训练集，验证集，测试集将两种语言分开
    train1 = "iwslt2016/de-en/train.tags.de-en.de"# 德语
    train2 = "iwslt2016/de-en/train.tags.de-en.en"# 英语
    eval1 = "iwslt2016/de-en/IWSLT16.TED.tst2013.de-en.de.xml"
    eval2 = "iwslt2016/de-en/IWSLT16.TED.tst2013.de-en.en.xml"
    test1 = "iwslt2016/de-en/IWSLT16.TED.tst2014.de-en.de.xml"
    test2 = "iwslt2016/de-en/IWSLT16.TED.tst2014.de-en.en.xml"
    for f in (train1, train2, eval1, eval2, test1, test2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip() for line in open(x, 'r',encoding='utf-8').read().split("\n") \
                      if not line.startswith("<")]
    prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    assert len(prepro_train1)==len(prepro_train2), "Check if train source and target files match."

    # eval
    _prepro = lambda x: [re.sub("<[^>]+>", "", line).strip() \
                     for line in open(x, 'r',encoding='utf-8').read().split("\n") \
                     if line.startswith("<seg id")]
    prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."

    # test
    prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."

    logging.info("Let's see how preprocessed data look like")
    #logging.info("prepro_train1: ", prepro_train1[0])
    # not all arguments converted during string formatting
    logging.info("prepro_train1: %s", prepro_train1[0])
    logging.info("prepro_train2: %s", prepro_train2[0])
    logging.info("prepro_eval1: %s", prepro_eval1[0])
    logging.info("prepro_eval2: %s", prepro_eval2[0])
    logging.info("prepro_test1: %s", prepro_test1[0])
    logging.info("prepro_test2: %s", prepro_test2[0])

    logging.info("# write preprocessed files to disk")
    os.makedirs("iwslt2016/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w',encoding='utf-8') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "iwslt2016/prepro/train.de")
    _write(prepro_train2, "iwslt2016/prepro/train.en")
    _write(prepro_train1+prepro_train2, "iwslt2016/prepro/train")
    _write(prepro_eval1, "iwslt2016/prepro/eval.de")
    _write(prepro_eval2, "iwslt2016/prepro/eval.en")
    _write(prepro_test1, "iwslt2016/prepro/test.de")
    _write(prepro_test2, "iwslt2016/prepro/test.en")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("iwslt2016/segmented", exist_ok=True)
    train = '--input=iwslt2016/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=iwslt2016/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("iwslt2016/segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w",encoding='utf-8') as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "iwslt2016/segmented/train.de.bpe")
    _segment_and_write(prepro_train2, "iwslt2016/segmented/train.en.bpe")
    _segment_and_write(prepro_eval1, "iwslt2016/segmented/eval.de.bpe")
    _segment_and_write(prepro_eval2, "iwslt2016/segmented/eval.en.bpe")
    _segment_and_write(prepro_test1, "iwslt2016/segmented/test.de.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("iwslt2016/segmented/train.de.bpe",'r',encoding='utf-8').readline())
    print("train2:", open("iwslt2016/segmented/train.en.bpe", 'r',encoding='utf-8').readline())
    print("eval1:", open("iwslt2016/segmented/eval.de.bpe", 'r',encoding='utf-8').readline())
    print("eval2:", open("iwslt2016/segmented/eval.en.bpe", 'r',encoding='utf-8').readline())
    print("test1:", open("iwslt2016/segmented/test.de.bpe", 'r',encoding='utf-8').readline())

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")
    
    
#-----------------BPE-----------------
# BPE（Byte Pair Encoding，双字节编码）
"""
BPE，（byte pair encoder）字节对编码，也可以叫做digram coding双字母组合编码，
主要目的是为了数据压缩，
算法描述为字符串里频率最常见的一对字符被一个没有在这个字符中出现的字符代替的层层迭代过程。
具体在下面描述。
该算法首先被提出是在Philip Gage的C Users Journal的 
1994年2月的文章“A New Algorithm for Data Compression”。

算法过程
   这个算法个人感觉很简单，下面就来讲解下：

   比如我们想编码：
          aaabdaaabac

   我们会发现这里的aa出现的词数最高（我们这里只看两个字符的频率），
   那么用这里没有的字符Z来替代aa：
           ZabdZabac
           Z=aa

   此时，又发现ab出现的频率最高，那么同样的，Y来代替ab：

           ZYdZYac
           Y=ab
           Z=aa

   同样的，ZY出现的频率大，我们用X来替代ZY：

           XdXac
           X=ZY
           Y=ab
           Z=aa

   最后，连续两个字符的频率都为1了，也就结束了。就是这么简单。
   解码的时候，就按照相反的顺序更新替换即可。

"""