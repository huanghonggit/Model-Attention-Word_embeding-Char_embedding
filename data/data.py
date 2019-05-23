import sys
sys.path.append(["../../", "../", "./"])
import torch
from text_utils import TextUtils
import numpy as np
from collections import Counter
from torch.autograd import Variable
from data.vocab_1 import WordVocab, CharVocab

# def to_categorical(y, num_classes=None, dtype='float32'):
#     y = np.array(y, dtype='int')
#     input_shape = y.shape
#     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#         input_shape = tuple(input_shape[:-1])
#     y = y.ravel()   # 多维变一维（返回视图）
#     if not num_classes:  # num_classes == None
#         num_classes = np.max(y) + 1
#     n = y.shape[0]  # 元素个数
#     categorical = np.zeros((n, num_classes), dtype=dtype)
#     categorical[np.arange(n), y] = 1
#     output_shape = input_shape + (num_classes,)
#     categorical = np.reshape(categorical, output_shape)
#     return categorical


class Instance(object):
    def __init__(self, words, tag):
        self.words = words  # 单词序列（经过分词）
        self.tag = tag      # 标签

    def __str__(self):
        return str(self.tag) + '\t' + ' '.join(self.words)  # 格式eg：1\t...


# 预测：对传入的数据列表进行预处理
def preprocess_data(input_list):
    insts = []
    for line in input_list:
        line = TextUtils.remove_blank(line)
        token_seq = TextUtils.tokenize(line)
        insts.append(Instance(token_seq, tag=-1))
    return insts


# 以Instance对象的形式保持数据和标签
def load_data_instance(file):
    insts = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as fin: # errors -- 设置不同错误的处理方案。默认为 'strict',意为编码错误引起一个UnicodeError。 其他可能得值有 'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace' 以及通过 codecs.register_error() 注册的任何值。
        for line in fin:
            label, text = line.strip().split('|||')
            insts.append(Instance(text.strip().split(), int(label.strip())))
    np.random.shuffle(insts)  # 初步随机化
    return insts



# 根据训练集数据创建WordVocab
def createWordVocab(corpus_path, lexicon_path=None):
    wds_counter = Counter() # 用counter()的目的是为了去掉出现次数太少的词；把；。等符号也进行了计数
    tags_counter = Counter()  # 计数（0,1,2）
    insts = load_data_instance(corpus_path)
    # 统计语料中词频和标签数量
    for inst in insts:
        for wd in inst.words:
            wds_counter[wd] += 1
        tags_counter[inst.tag] += 1

    return  WordVocab(wds_counter, tags_counter, lexicon_path)#Vocab(wds_counter, tags_counter, lexicon_path)

# 根据训练集数据创建CharVocab #############
def createCharVocab(): # corpus_path, lexicon_path=None
    return  CharVocab()
# instance数据转id索引
# def data_to_index(instances, wd_to_idx, lexicon=None):
#     assert instances is not None
#     wd_insts = instances
#     for i, inst in enumerate(wd_insts):
#         wids, att_ids = [], []
#         for wd in inst.words:
#             if wd in wd_to_idx.keys():
#                 wids.append(wd_to_idx[wd])
#             else:
#                 wids.append(0)
#
#             if lexicon is not None:
#                 if wd in lexicon:  # 注意力监督：序列中存在于情感词典中的词，对应位置为1
#                     att_ids.append(1)
#                 else:
#                     att_ids.append(0)
#
#         if lexicon is not None:
#             wd_insts[i].extra = att_ids
#
#         wd_insts[i].words = wids
#
#     return wd_insts

# 产生batch
def get_batch(data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(data)

    nb_batch = int(np.ceil(len(data) / float(batch_size)))
    for i in range(nb_batch):
        batch_data = data[i*batch_size: (i+1)*batch_size]
        yield batch_data


# 对齐batch数据
# def pad_batch(batch_data, max_len, pad=0):
#     pad_batch_data = np.array(batch_data)
#     seqs_len_lst = []
#     # 取预设最大序列长度和实际最大长度的最小值
#     max_len = min(max_len, max([len(inst.words) for inst in batch_data]))
#     for i, inst in enumerate(pad_batch_data):
#         if len(inst.words) > max_len:
#             seqs_len_lst.append(max_len)
#
#             inst.words = inst.words[: max_len]
#         elif len(inst.words) < max_len:
#             seqs_len_lst.append(len(inst.words))
#
#             for j in range(max_len - len(inst.words)):
#                 inst.words.append(pad)
#         else:
#             seqs_len_lst.append(max_len)
#
#         pad_batch_data[i] = inst
#
#     # 对序列长度从大到小排列，indices存对应原始序列的索引
#     sorted_seqs_len, indices = torch.sort(torch.tensor(seqs_len_lst), descending=True)  # 默认升序
#
#     _, unsorted_indices = torch.sort(indices)  # 恢复排序前的顺序
#
#     # padded_batch_data = torch.index_select(torch.tensor(pad_batch_data), 0, indices)
#
#     if indices.numel() > 1:  # indices不止一个元素
#         pad_batch_data = pad_batch_data[indices]
#
#     # 返回排序后的序列和序列长度
#     return pad_batch_data, sorted_seqs_len.numpy(), unsorted_indices


# 将原始数据转换成Tensor
def batch_data_variable(batch_data, vocab, char_vocab):
    batch_size = len(batch_data)
    max_seq_len, max_wd_len = 0, 0
    # max_len = max([len(inst.words) for inst in batch_data])
    for inst in batch_data:
        if len(inst.words) > max_seq_len:
            max_seq_len = len(inst.words)
        for wd in inst.words:
            if len(wd) > max_wd_len:
                max_wd_len = len(wd)

    # 归零的初始化
    # corpus_idxs = Variable(torch.zeros(batch_size, max_len, dtype=torch.long), requires_grad=False)
    wd2vec_idxs = Variable(torch.zeros(batch_size, max_seq_len, dtype=torch.long), requires_grad=False) # 每句话
    char_idxs = Variable(torch.zeros(batch_size, max_seq_len, max_wd_len, dtype=torch.long), requires_grad=False) # 每句话，每个词，每个字
    att_ids = Variable(torch.zeros(batch_size, max_seq_len, dtype=torch.long), requires_grad=False)
    tags = torch.zeros(batch_size, dtype=torch.long)

    seq_lens = []
    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        seq_lens.append(seq_len)
        wd2vec_idxs[i, :seq_len] = torch.LongTensor(vocab.word2index(inst.words))
        tags[i] = vocab.tag2index(inst.tag)
        att_ids[i, :seq_len] = torch.LongTensor(vocab.lexicon_vec(inst.words))
        for j, wd in enumerate(inst.words):# 对每个词进行单个汉字embedding
            char_idxs[i, j, :len(wd)] = torch.LongTensor(char_vocab.char2idx(wd))
        # corpus_idxs[i, :seq_len] = torch.LongTensor(vocab.corpus_wd2idx(inst.words))


    # 对序列长度从大到小排列，indices存对应原始序列的索引
    sorted_seq_lens = [] # 保存的是按长度的降序进行排序，保存的每句话长度值
    sorted_seq_lens, indices = torch.sort(torch.tensor(seq_lens), descending=True)  # 默认升序,这里是使用降序，indices是原始输入、排序前的下标
    _, unsorted_indices = torch.sort(indices)  # 恢复排序前的序列顺序
    # 调整序列顺序
    # corpus_idxs = torch.index_select(corpus_idxs, dim=0, index=indices)
    wd2vec_idxs = torch.index_select(wd2vec_idxs, dim=0, index=indices)
    tags = torch.index_select(tags, dim=0, index=indices)
    att_ids = torch.index_select(att_ids, dim=0, index=indices)

    return wd2vec_idxs, tags, att_ids, char_idxs, sorted_seq_lens # , unsorted_indices   corpus_idxs,


# -------------------------------------------------------------------
# 加载训练语料，格式：标签  词序列(分好词的)
# def load_data(file):
#     xs, ys = [], []
#     with open(file, 'r') as fin:
#         for example in fin:
#             lbl, sent = example.split('\t')
#             xs.append(sent.split())
#             ys.append(int(lbl))
#     xs, ys = np.array(xs), np.array(ys)
#     assert len(xs) == len(ys)
#     # shuffle_ids = np.arange(len(ys))
#     # np.random.shuffle(shuffle_ids)
#     shuffle_ids = np.random.permutation(len(ys))  # 随机化序列
#     xs, ys = xs[shuffle_ids], ys[shuffle_ids]
#     return xs, ys

# 分词序列转索引
# def seqs_to_ids(seqs, wd_to_idx):
#     xs = []
#     for wd_seq in seqs:
#         wids = []
#         for wd in wd_seq:
#             if wd in wd_to_idx.keys():
#                 wids.append(wd_to_idx[wd])
#             else:
#                 wids.append(0)
#         xs.append(wids)
#     return np.array(xs)

# 对齐索引序列，默认pad = 0
# def pad_sequences(seqs, maxlen=0, value=0, padding='post', truncating='post'):
#     if not isinstance(seqs, np.ndarray):
#         seqs = np.array(seqs)
#
#     z = value * np.ones((len(seqs), maxlen))
#     for i, seq in enumerate(seqs):
#         seq_len = len(seq)
#         if seq_len > maxlen:  # 截取
#             if truncating == 'pre':
#                 z[i] = seq[-maxlen:]
#             elif truncating == 'post':
#                 z[i] = seq[:maxlen]
#             else:
#                 raise ValueError('Truncating type %s not understood' % truncating)
#         elif seq_len < maxlen:  # 补齐
#             if padding == 'pre':
#                 z[i, -seq_len:] = seq
#             elif padding == 'post':
#                 z[i, :seq_len] = seq
#             else:
#                 raise ValueError('Padding type %s not understood' % padding)
#         else:
#             z[i] = seq
#     return z

# 对齐索引序列，返回按长度降序排列的对齐索引序列，排序索引，降序的长度列表
# def pad_sequences_rnn(seqs, maxlen=0, value=0, padding='post', truncating='post'):
#     seq_len_lst = []
#     if not isinstance(seqs, np.ndarray):
#         seqs = np.array(seqs)
#     Xs = value * np.ones((len(seqs), maxlen))
#     for i, seq in enumerate(seqs):
#         seq_len = len(seq)
#         if seq_len > maxlen:  # 截取
#             seq_len_lst.append(maxlen)
#
#             if truncating == 'pre':
#                 Xs[i] = seq[-maxlen:]
#             elif truncating == 'post':
#                 Xs[i] = seq[:maxlen]
#             else:
#                 raise ValueError('Truncating type %s not understood' % truncating)
#         elif seq_len < maxlen:  # 补齐
#             seq_len_lst.append(seq_len)
#
#             if padding == 'pre':
#                 Xs[i, -seq_len:] = seq
#             elif padding == 'post':
#                 Xs[i, :seq_len] = seq
#             else:
#                 raise ValueError('Padding type %s not understood' % padding)
#         else:
#             seq_len_lst.append(seq_len)
#
#             Xs[i] = seq
#
#     # 对序列长度进行降序，indices保持原序列的索引
#     sorted_seqs_len, indices = torch.sort(torch.IntTensor(seq_len_lst), descending=True)
#     Xs = Xs[indices]
#     return Xs, indices.numpy(), sorted_seqs_len.numpy()
