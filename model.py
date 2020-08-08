# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1) batch_size,time_steps
            # source的mask
            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            # enc += positional_encoding(enc, self.hp.maxlen1) 
            #位置编码的维度和词向量的维度一致才能相加
            # 对于topic to essay 这个地方感觉不需要位置向量
            # 编码器部分不需要位置向量但是解码器部分需要位置向量
            
            #词向量也进行dropout
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention because queries,keys and values both are 'enc'
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # 这个causality决定是否对未来的词进行mask，False则可以看到未来的词
                    # feed forward + residual connection
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32 #model预测的id序列
        y: (N, T2). int32 # ground truth id sequence
        sents2: (N,). string. # gournd truth token sequence
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)
            # target的mask
            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    # because queries,keys and values both are 'dec'
                    # 这个之所以叫做self attention 是因为queries,keys,valus都来自decoder
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")
                    # 这个self-attention需要mask掉后面的token
                    # Vanilla attention
                    # 这个和rnn的attention非常像
                    # queries来自decoder ,keys和values来自encoder
                    # queries from decoder
                    # keys and values from encoder
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward + residual connection
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        # 可以实现多维tensor和低维tensor的直接相乘，是一种便捷的矩阵运算方法
        # dec (N,T2,d_model) *(d_model,vocab_size)->(N, T2, vocab_size)
        # 对应 'ntd,dk->ntk'
        """
        # Matrix multiplication
        >>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]

        # Dot product
        >>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

        # Outer product
        >>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]

        # Transpose
        >>> einsum('ij->ji', m)  # output[j,i] = m[i,j]

        # Batch matrix multiplication
        >>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
        
        """
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))# (N, T2)

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        # 参看论文5.4，这会降低困惑，因为模型学习会更加不确定，提高了准确性和BLEU分数
        # 这个技巧可以尝试
        
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<PAD>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def top_k_logits(self, logits, k):
        if k == 0:
            # no truncation
            return logits

        def _top_k():
            values, _ = tf.nn.top_k(logits, k=k)
            min_values = values[:, -1, tf.newaxis]
            return tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )
        return tf.cond(
           tf.equal(k, 0),
           lambda: logits,
           lambda: _top_k(),
        )


    def top_p_logits(self,logits, p):
        """Nucleus sampling"""
        batch, length = logits.shape.as_list()
        #sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
        #tensorflow 1.12.0中没有tf.sort只能使用下面的替代
        print("logits shape :",logits.shape)
        sorted_logits,_= tf.nn.top_k(logits, k=length,sorted=True)
        print("sorted_logits",sorted_logits)
        # tf.cumsum累计求和维度不变
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        print("cumulative_probs  :",cumulative_probs )
        indices = tf.stack([
            tf.range(0, batch),
            # number of indices to include
            tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
        ], axis=-1)
        # 收集概率和为p的边界点
        print("indices :",indices)
        min_values = tf.gather_nd(sorted_logits, indices)
        print("min_values :",min_values)
        return tf.where(
            logits < min_values,
            tf.ones_like(logits) * -1e10,
            logits,
        )

    # tensorflow 1.13.0
    def eval_smaple(self, xs, ys,top_k=100,top_p=1,temperature = 0.5):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<GO>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):#解码过程
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            logits = logits[:, -1, :]  / tf.to_float(temperature)
            
            logits = self.top_k_logits(logits, k=top_k)
            print("logits  is :",logits)
            logits = self.top_p_logits(logits, p=top_p)
            y_hat = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<PAD>"]: break
            #decoder_input 之前解码的token ids#
            #[batch_size,t]  y_hat是当前解码的token id
            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        #这里的n是采样到summary中的
        # tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))返回6*6的矩阵，
        # 产生于low和high之间，产生的值是均匀分布的。
        sent1 = sents1[n] #source sentence
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)#model sentence
        sent2 = sents2[n]#targtet sentence

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()
        
        return pred,summaries
        #return y_hat, summaries

    def eval_one(self, input_tokens):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        """
        xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)

        ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
        """
        dict = self.token2idx
        x = [[dict.get(t, dict["<UNK>"]) for t in input_tokens]]
        x_seqlens = len(x[0])
        sent1 = convert_idx_to_token_tensor(x[0], self.idx2token)
        xs = (x,x_seqlens,sent1)
        ys = (tf.zeros([self.hp.batch_size,self.hp.maxlen2],dtype=tf.int32),
        tf.zeros([self.hp.batch_size,self.hp.maxlen2],dtype=tf.int32),
        tf.zeros([self.hp.batch_size],dtype=tf.int32),
        tf.zeros([self.hp.batch_size],dtype=tf.string),
        )
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<GO>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):#解码过程
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<PAD>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

       
        pred = convert_idx_to_token_tensor(y_hat[0], self.idx2token)#model sentence
        

        return pred
     
    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<GO>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):#解码过程
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<PAD>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        # tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))返回6*6的矩阵，
        # 产生于low和high之间，产生的值是均匀分布的。
        sent1 = sents1[n] #source sentence
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)#model sentence
        sent2 = sents2[n]#targtet sentence

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

