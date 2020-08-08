# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''
# 对于需要考虑上下文的任务比如阅读理解，命名实体识别，文本分类，情感分类问题回答等等tranfomer比较有优势
# 但是对于文本生成由于我们无法考虑上下文，只能是流式生成所以tanrsformer效果不是很好

import numpy as np
import tensorflow as tf
#import seaborn as sns
#import matplotlib.pyplot as plt
def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:] #最后一个维度
        # # (N, T_q, d_model)
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True) 
        #keep_dims表示原来是几维张量，计算结果也是几维张量
        # (N, T_q, 1),(N, T_q, 1)
        # # 用于在指定维度计算均值与方差
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        # (N, T_q, d_model)  (N, T_q, 1)
        # epsilon使用一个比较小的数防止分母为0的情况
        outputs = gamma * normalized + beta
        # (d_model,)*(N, T_q, d_model) + (d_model,)
        # 对于'*'底维度必须相同，对高维度进行广播
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding 
    # n.因果关系; 因果律(或性); 
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        # Q-[N, T_q, d_k]  K-[N, T_k, d_k].
        #3维张量相乘，第一个维度相同，后两个维度满足矩阵相乘的法则

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs) #经过mask部分的概率 exp(-∞)->0
        # 当给mask矩阵为0的对应位置替换一个负很大的值后，相应attention的结果就会趋近为0。
        # 这个计算的就是注意力attenion
        attention = tf.transpose(outputs, [0, 2, 1]) # # (N, T_k,T_q)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # --------注意力机制的可视化包括传统的注意力机制和self attention-------------
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1)) # (N, T_k,T_q,1)
        # tensor：uint8或者float32型的4-D Tensor[batch_size, height, width, channels]，
        # 其中channels为1， 3 或者4。
        # 对attention进行可视化
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # --------注意力机制的可视化包括传统的注意力机制和self attention-------------
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #sns.heatmap(attention[:1])
        # 画彩色的注意力机制图
        
        #plt.show()
        """
        name：节点的名字，也就是在tensorboard上面会显示的名字。
        tensor：格式必须是四维的[batch_size,height, width, channels]，
        对于channels：
        channels=1为灰度图像
        channels=3为RGB图像
        channels=4为RGBA图像（Red（红色） Green（绿色） Blue（蓝色）和 Alpha合成，也代表了透明度）
    
        """
        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        # outputs-(N, T_q, T_k)  V-[N, T_k, d_v].

    return outputs


"""
mask是Transformer中很重要的一个概念，mask操作的目的有两个：
让padding(不够长补0)的部分不参与attention操作
生成当前词语的概率分布时，让程序不会注意到这个词背后的部分
上面的第一个目的分别对应的是普通的Scaled Dot-Product Attention中的mask操作，
而后一个目的对应的是Masked Multi-Head Attention中的Masked

"""

"""
Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。
padding mask 在所有的 scaled dot-product attention 里面都需要用到
sequence mask 只有在 decoder 的 self-attention 里面用到

"""
def mask(inputs, key_masks=None, type=None):
    # 这个mask的bert的mask还是不一样的
    
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N,  T_k)
    #key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1 #这是一个很大的负数 代替 -∞
    # 先进行key的mask，相当于找出key的padding，让它softmax后的概率为0，
    # 在计算context vector的时候，让其贡献为0
    # key和query都是mask掉pad的ids就是序列id pad为0的部分
    
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        # [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]
        # [h*N // N ,1] [h,1]
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        # “//”，在python中，这个叫“地板除”，3//2=1
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num  #padding的位置变为负无穷大，不padding的位置不变
    # 即要让query padding部分的时间步骤对K的attention全部为0，这样在计算context vector后才能为0
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     # tf.sign x<0 return -1 x= 0 return 0 x>0 return 1
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    #     (h*N, T_q, T_k)
    #
    #
    # 对未来信息进行mask，让self attention的时候看不到未来的词，即在计算context vector的时候，
    # 未来的词的概率为0，
    # 对计算context vector的贡献为0，这个只在transformer decoder端使用
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        # 这一句的意思是生成一个下三角矩阵，下三角矩阵用来对decoder的结果进行mask
        
        """
          [[1,0,0],
          [1,1,0],
          [1,1,1]]
        """
        # 第一个token的生成之和第一个有关看一行
        # 第二个token的生成之和前两个有关看第二行
        # 第三个token的生成之和前三个有关看第三行
        # tf.linalg.LinearOperatorLowerTriangular 可见这个函数把输入的张量转换为了下三角矩阵，
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        """
          [[-∞,-∞,-∞],
          [-∞,-∞,-∞],
          [-∞,-∞,-∞]]
        """
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        """
          [[0.2,-∞,-∞],
          [0.3,0.1,-∞],
          [0.3,0.3,0.2]]
        """
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  C = d_model
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        
        # Split and concat
        # h is the  num_heads
        # tf.split  h个[N,T_q,d_model/h]   # tf.concat (h*N, T_q, d_model/h)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training) # (h*N, T_q, d_model/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
        # h个张量的列表 (N,T_q,d_model/h) ，经过Session之后变为ndarray
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs) # (N, T_q, d_model)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    #主要是一个非线性变换的功能
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        # 两次全连接
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    """
    1) 第一部分：将原本Dirac分布的标签变量替换为(1 - ϵ)的Dirac函数；
    2) 第二部分：以概率 ϵ ，在u(k)
    中份分布的随机变量。
    对于损失函数，我们需要用预测概率去拟合真实概率，
    而拟合one-hot的真实概率函数会带来两个问题：1)无法保证模型的泛化能力，
    容易造成过拟合；2) 全概率和0概率鼓励所属类别和其他类别之间的差距尽可能加大，而由梯度有界可知，
    这种情况很难adapt。会造成模型过于相信预测的类别。
    """
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    # V是词典大小32,000
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
        # 1,T -> N T
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])
        # 对于上面的i=2k i-i%2=2k
        # i = 2k+1 i-i%2 =2k  
        # 所以i-i%2始终为偶数
        
        #position_enc = pos/10000^(2i/d_model)
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        # pe(pos,2i) = sin(position_enc)
        # [:,0::2]表示从第一个元素开始以2step遍历0, 2, 4, 6,...
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        # pe(pos,2i+1) =cos(position_enc)
        # [:,1::2]表示从第二个元素开始以2step遍历1, 3, 5, 7,...
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)
        # 翻译过来：这个函数把python的变量类型转换成tensor，而这个value可以是tensor，
        # numpy arrays(numpy 的数组)，python list（python 列表）python的变量
        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    # 学习速率衰减
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)