import numpy as np
import time


class GenDataLoader(object):
    def __init__(self, batch_size, source_index, source_len, target_idx, target_len,
                 max_len, source_label=None, memory=None):
        assert len(source_index) == len(target_idx)
        self.batch_size = batch_size
        self.source_idx = source_index
        self.source_len = source_len
        self.target_idx = target_idx
        self.target_len = target_len
        self.max_len = max_len
        self.has_label = False
        self.has_mem = False
        if source_label is not None:
            self.has_label = True
            self.source_label = source_label
        if memory is not None:
            self.has_mem = True
            self.memory = memory
        #self.num_batch = len(source_index) // batch_size 
        self.num_batch = len(source_index)  // batch_size


    def create_batch(self):
        self.si_batch = np.split(self.source_idx[:self.num_batch * self.batch_size], self.num_batch)
        self.sl_batch = np.split(self.source_len[:self.num_batch * self.batch_size], self.num_batch)
        self.tl_batch = np.split(self.target_len[:self.num_batch * self.batch_size], self.num_batch)
        self.ti_batch = np.split(self.target_idx[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_label: #是否有话题标签
            self.slbl = np.split(self.source_label[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_mem: # 是否有记忆
            self.smem = np.split(self.memory[:self.num_batch * self.batch_size], self.num_batch)

        self.g_pointer = 0

    def next_batch(self):
        generator_batch = [self.si_batch[self.g_pointer],
                           self.sl_batch[self.g_pointer],
                           self.ti_batch[self.g_pointer],
                           self.tl_batch[self.g_pointer],
                           ]
        if self.has_label:
            generator_batch.append(self.slbl[self.g_pointer])
        if self.has_mem:
            generator_batch.append(self.smem[self.g_pointer])
        self.g_pointer = (self.g_pointer + 1) % self.num_batch
        return generator_batch

    def reset_pointer(self):
        self.g_pointer = 0



def shuffle_data(num, *data):
    size = len(data[0])
    permutation = np.random.permutation(size)
    ret = []
    for d in data:
        d = d[permutation]
        ret.append(d[:num])
    return ret


def padding(index, max_len):
    batch_size = len(index)
    padded = np.zeros([batch_size, max_len])
    for i, seq in enumerate(index):
        for j, element in enumerate(seq):
            padded[i, j] = element
    return padded


def get_weights(lengths, max_len):
    x_len = len(lengths)
    ans = np.zeros((x_len, max_len))
    for ll in range(x_len):
        kk = lengths[ll] - 1
        for jj in range(kk):
            # print(ll)
            # print(jj)
            ans[ll][jj] = 1 / float(kk)
    return ans

# 使用numpy进行训练集和测试集的划分

def prepare_data(test_ratio, *data):
    length = len(data[0])
    test_size = int(length * test_ratio)
    print(test_size)
    print(length - test_size)
    permute = np.random.permutation(length)
    train = []
    test = []
    for d in data:
        d = d[permute]
        d_test = d[:test_size]
        d_train = d[test_size:]
        train.append(d_train)
        test.append(d_test)
    return train, test


def load_npy(data_config):
    ret = []
    # print(data_config)
    for item in data_config:
        # print(item)
        ret.append(np.load(item))
    return ret


def to_one_hot(arr, num_class):
    size = len(arr)
    lbl = np.zeros([size, num_class])
    for i in range(size):
        lbl[i, arr[i]] += 1
    return lbl

