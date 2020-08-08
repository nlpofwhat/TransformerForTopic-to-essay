from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from collections import defaultdict
import numpy as np
from tqdm import tqdm
# nltk has to be installed
# build a lis

#test_src
test_src = 'data/test_src.txt'
target = 'data/test_tgt.txt'
pre = 'data/zhihu100L3.34-41700B0'
#pre = 'data/zhihu108L3.29-45036B0.80'
vocab_fpath = 'data/word_dict_zhihu.npy'
token2idx = np.load(vocab_fpath).item()
#print(token2idx)
#这个npy是一个字典，word的列表的形式
#vocab_dict = {w:id for id,w in enumerate(vocab_dict)}
# 这个npy是一个字典，{word:id of word}的形式
idx2token = {v: k for k, v in token2idx.items()}
test_samples = []
test_target = []
topic_list = []
num = 0
total_bleu = 0
g = []
with open(test_src,'r',encoding='utf-8') as f1,\
    open(target,'r',encoding='utf-8') as f2,\
    open(pre,'r',encoding='utf-8') as f3:
    
        for x,y,y_pre in zip(f1,f2,f3):
            if num<2240:
                num = num+1
                y_pre = y_pre.strip().split(' ')
                test_samples.append(y_pre)
                y = y.strip().split(' ')
                g.append(y)
                test_target.append(y)
                x = x.strip().split(' ')
                topic_list.append([token2idx[x_] for x_ in x]) 
            else:
                break
tp = [sorted(x) for x in topic_list]  # sort topic word
#tp = [x for x in topic_list]
tw = list(map(lambda x: " ".join([idx2token[w] for w in x]), tp))
#if self.refers is None:
print("building refers ....")
multi_refers = defaultdict(list)

for w, r in zip(tw, test_target):
    multi_refers[w].append(r)
selfrefers = multi_refers

max_bleu = 0
b_t = None
b_r = None
b_g = None
get_ret = False

refer = []
for w, h in zip(tw, test_samples):
    # print(w)
    refers = selfrefers[w]  # 真实的句子
    #print(h)
    refer.append(h)
    if len(refers) == 0:
        raise Exception("Error")
    # h = [i for i in h if i!=1 ]
    total_bleu += sentence_bleu(refers, h, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
    cur_bleu = sentence_bleu(refers, h, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)
    
    if cur_bleu > max_bleu:
        max_bleu = cur_bleu
        b_t = w
        b_r = refers[0]
        b_g = h
print(len(tw))
print("bleu : ",total_bleu / len(tw) * 100)
# return total_bleu / dataloader.num_batch


def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict

def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count

def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
        #if freq == 1:
        #    ngram_distinct_count += freq
    return (ngram_distinct_count + 0.0001) / (ngram_total + 0.0001)

def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    return [distinct1, distinct2]
    
    
    
    
    
def ngram(s, n):

    return list(zip(*(s[i:] for i in range(n))))
def _calculate_distinct_n(hypotheses, n):

    ngrams = ngram(hypotheses, n)
    
    #ngrams = [x[0] for x in ngrams]
    #print(ngrams)
    n_grams = len(ngrams)
    unique_ngrams = np.unique(["".join(list(x)) for x in ngrams])
    n_unique_ngrams = len(unique_ngrams)

    return n_unique_ngrams / n_grams if n_grams > 0 else 0
def calculate_distinct_1(hypotheses):

    return _calculate_distinct_n(hypotheses, 1)

def calculate_distinct_2(hypotheses):

    return _calculate_distinct_n(hypotheses, 2)

def Jaccrad(model, reference):
    #terms_reference为源句子，terms_model为候选句子    
    #terms_reference= jieba.cut(reference)#默认精准模式    
    #terms_model= jieba.cut(model)    
    grams_reference = reference#去重；如果不需要就改为list    
    grams_model = model   
    temp=0    #统计交集的个数
    for i in grams_reference:        
        if i in grams_model:            
            temp=temp+1    
    fenmu=len(grams_model)+len(grams_reference)-temp #统计并集的个数并集    
    jaccard_coefficient=float(temp/fenmu)#交集    
    return jaccard_coefficient 


h = test_samples#生成的文本
"""
sents = []
for x,y in zip(h,g):
    sents.append([x,y])

dis_1,dis_2 = calc_distinct(sents)
print("dis_1:",dis_1*100)
print("dis_2:",dis_2*100) 
"""
# 1.51  6.91


dis_1 = 0
dis_2 = 0
for h_ in h:                
    dis_1 += calculate_distinct_1(h_)
    dis_2 += calculate_distinct_2(h_) 
print("generated")    
print("dis_1:",dis_1/len(h)*100)
print("dis_2:",dis_2/len(h)*100)   

"""
#Diversity(S_i) = 1 - max\{J(S_i,S_j)\}_{j=1}^{j=|S|,i\neq{j}}.
diversity = 0
for i,h_1 in tqdm(enumerate(h)):
    #diversity
    max_d = 0
    for h_2 in h:
        #print(h_1,h_2)
        if h_1!=h_2:
            tmp = Jaccrad(h_1,h_2)
            if tmp>max_d:
                max_d = tmp
    diversity+=(1-max_d)
print("diversity:",diversity/len(h)*100)

#Novelty(S_i) = 1 - max\{J(S_i,C_j)\}_{j=1}^{j=|C|}.
Novelty = 0
for i,h_1 in tqdm(enumerate(h)):#生成的句子
    #diversity
    max_d = 0
    for h_2 in g:
        #print(h_1,h_2)
        if h_1!=h_2: #训练集的句子
            tmp = Jaccrad(h_1,h_2)
            if tmp>max_d:
                max_d = tmp
    Novelty+=(1-max_d)
print("Novelty:",Novelty/len(h)*100)
"""