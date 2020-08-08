class Config(object):
    training_config = {
        "tat_path": "model/tat/",
        "baseline_epoch": 80,#80
        "tav_path": "model/tav/",
        "mta_path": "model/mta/",
        "sclstm_path": "model/sc_lstm/",
        "htm_path": "model/htm/mta_hm_ex",
        "htmvl_path": "model/htmvl/",
        "mem_path": "model/mem/",
        "word_dict": "./data/word_dict_zhihu.npy",  #字典
        "pretrain_wv": "./data/wv_tencent.npy"   #预训练的词向量

    }
    
    #train_data_path = 'data/small_train.txt'   # 训练数据的路径，相对当前路径
    #test_data_path = 'data/small_train.txt'   # 测试数据的路径，两者的数据格式一样的
    
    
    train_data_path_zhihu = [
        # "si_train":
        "./data/train_src.npy",  # 话题ids
        # "sl_train":   
        "./data/train_src_len.npy", # 话题的长度
        # "s_lbl_train":
        #"./data/train_src_lbl_oh.npy", #
        # "ti_train":
        "./data/train_tgt.npy", # 句子ids
        # "tl_train":
        "./data/train_tgt_len.npy" #,# 句子实际长度
        # memory
        #"./data/train_mem_idx_120_concept.npy" #外部知识三元组
    ]

    test_data_path_zhihu = [
        # "si_train":
        "./data/tst.src.npy",
        # "sl_train":
        "./data/tst.src.len.npy",
        # "s_lbl_train":
        #"./data/tst.src.lbl.oh.npy",
        # "ti_train":
        "./data/tst.tgt.npy",
        # "tl_train":
        "./data/tst.tgt.len.npy",
        # memory
        "./data/tst.mem.idx.120.concept.npy"
    ]

    val_data_path_zhihu = [
        # "si_train":
        "./data/val.src.npy",
        # "sl_train":
        "./data/val.src.len.npy",
        # "s_lbl_train":
        #"./data/val.src.lbl.oh.npy",
        # "ti_train":
        "./data/val.tgt.npy",
        # "tl_train":
        "./data/val.tgt.len.npy"#,
        # memory
        #"./data/val.mem.idx.120.concept.npy"
    ]
    
    """

    train_data_path = [
        # please use your own preprocessed data. 
    ]   #已经预处理好的数据
    test_data_path = [
        # please use your own preprocessed data. 
    ] #已经预处理好的数据
    """

    generator_config_zhihu = {
        "embedding_size": 200,  # tencent 200 dim
        "hidden_size": 512,
        "max_len": 100,   #这个了max_len应该和判别器中的长度一致
        "start_token": 0,
        "eos_token": 1,
        "batch_size": 64,
        "vocab_size": 50004,
        "grad_norm": 10,
        "topic_num": 5,
        "is_training": True,
        "keep_prob": .5,
        "norm_init": 0.05,
        "normal_std": 1,
        "learning_rate": 1e-3,
        "beam_width": 5,
        "mem_num": 120,  # 注意max_len和mem_num的关系是怎样的
        "attention_size": 512 #
    }


