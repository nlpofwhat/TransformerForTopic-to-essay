# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)
    """
    这个函数可以将任意的python函数转化为tensorflow op
    my_func 是任意的python函数
    inputs是
    func: 一个 Python 函数, 它接受 NumPy 数组作为输入和输出，
    并且数组的类型和大小必须和输入和输出用来衔接的 Tensor 大小和数据类型相匹配.
    inp: 输入的 Tensor 列表.
    Tout: 输出 Tensor 数据类型的列表或元祖.
    
    这是一个可以把 TensorFlow 和 Python 原生代码无缝衔接起来的函数，
    有了它，你就可以在 TensorFlow 里面自由的实现你想要的功能，
    而不用考虑 TensorFlow 有没有实现它的 API，
    并且可以帮助我们实现自由的检查该功能模块的输入输出是否正确，
    而不受到TensorFlow 的先构造计算图再运行导致的不能单独检测单一模块的功能的限制； 
    
    输出也是numpy array，也可以有多个输出。inp传入输入值，Tout指定输出的基本数据类型。
    """
# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry


# 解码语句的后处理过程去掉一些特殊的标记符号e.g.<GO>/<START>/<END>/<EOS>/<S>/</S>
def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = " ".join(idx2token[idx] for idx in h) # 每个词使用空格分开
        sent = sent.split("<EOS>")[0].strip() 
        #[0]表示解码到</s>符号为止，并且去掉解码可能产生多余的空格
        #sent = sent.replace("▁", " ") # remove bpe symbols，这是bpe算法得到的标记
        # 中文分词没有这个所以不需要
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d) #参数以字典的形似保存
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path

    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)# run script in cmd
    
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score) #新的文件名加bleu
        os.system("mv {} {}".format(translation, new_translation))# linux命令更改文件名
        """
        mv 文件名 文件名  将源文件名改为目标文件名
        mv 文件名 目录名  将文件移动到目标目录
        mv 目录名 目录名  目标目录已存在，将源目录移动到目标目录；目标目录不存在则改名
        mv 目录名 文件名  出错
        """
        os.remove(translation)

    except: pass
    os.remove("temp")


# def get_inference_variables(ckpt, filter):
#     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
#     return vars




