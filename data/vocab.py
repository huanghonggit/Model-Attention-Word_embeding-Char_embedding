# from collections import Counter, defaultdict
import gensim
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pickle


# 词表有2个：一个是来自语料中的词表，一个是来着word2vec词向量中的词表
class Vocab(object):
    def __init__(self, wds_counter, tags_counter, lexicon_path=None):
        self.UNK = 0
        self.UNK_TAG = -1
        self.min_count = 5
        self._word2index = {}
        self._index2word = {}
        self._lexicon = set()

        self._wd2freq = {wd: count for wd, count in wds_counter.items() if count > self.min_count} # 输出是词典{word：count}，去掉出现次数小于2 的词

        self._corpus_wd2idx = {wd: idx+1 for idx, wd in enumerate(self._wd2freq.keys())}  # enumerate idx是从0开始   从word：1开始
        self._corpus_wd2idx['<unk>'] = self.UNK  # 针对oov一对{k:v}
        self._corpus_idx2wd = {idx: wd for wd, idx in self._corpus_wd2idx.items()}  # {idx : word}

        self._tag2idx = {tag: idx for idx, tag in enumerate(tags_counter.keys())}  # tags_counter里面是tag（0,1,2）和对应的count； tag:0开始
        self._idx2tag = {idx: tag for tag, idx in self._tag2idx.items()}

        if lexicon_path is not None:
            # 加载情感词典，格式：一行一个情感词
            with open(lexicon_path, 'r', encoding='utf-8', errors='ignore') as fin:
                for wd in fin:
                    wd = wd.strip()
                    if wd != '':
                        self._lexicon.add(wd)

            print('lexicon size:', len(self._lexicon))

    # 获得embedding权重向量和词索引表
    #     def get_embedding_weights(self, vocab_path):
    #         wd2vec_model = Word2Vec.load(vocab_path)
    #         if wd2vec_model is not None:
    #             gensim_dict = Dictionary()  # {索引: 词}
    #             # 实现词袋模型
    #             gensim_dict.doc2bow(wd2vec_model.wv.vocab.keys(), allow_update=True)  # (token_id, token_count)
    #             self._word2index = {wd: idx + 1 for idx, wd in gensim_dict.items()}  # 词索引字典 {词: 索引}，索引从1开始计数
    #             self._word2index['<unk>'] = self.UNK
    #             self._index2word = {idx: wd for wd, idx in self._word2index.items()}
    #             # word_vectors = {wd: wd2vec_model.wv[wd] for wd in self._word2index.keys()}  # 词向量 {词: 词向量}
    #
    #             vocab_size = len(self._word2index)  # 词典大小(索引数字的个数)
    #             embedding_weights = np.zeros((vocab_size, wd2vec_model.vector_size))  # vocab_size * EMBEDDING_SIZE的0矩阵
    #             for idx, wd in self._index2word.items():  # 从索引为1的词语开始，用词向量填充矩阵
    #                 # embedding_weights[idx, :] = word_vectors[wd]
    #                 if idx != self.UNK:
    #                     word_vector = wd2vec_model.wv[wd]
    #                     embedding_weights[idx] = word_vector
    #                     # embedding_weights[self.UNK] += word_vector
    #
    #             # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
    #             # embedding_weights[self.UNK, :] = np.random.uniform(-0.25, 0.25, config.embedding_size)
    #             # embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, config.embedding_size)
    #             # embedding_weights[self.UNK] = embedding_weights[self.UNK] / vocab_size
    #
    #             return embedding_weights
    # 方法二
    # 获得embedding权重向量和词索引表
    def get_embedding_weights(self, word2vec_path, embedding_size):
        # wd2vec_model = gensim.models.KeyedWord2Vec.load(vocab_path)
        # 直接使用预训练词向量
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
        if wvmodel is not None:
            gensim_dict = Dictionary()  # {索引: 词}
            # 实现词袋模型
            gensim_dict.doc2bow(wvmodel.wv.vocab.keys(), allow_update=False)  # (token_id, token_count);当使用预训练词向量时，不需要更新词向量
            self._word2index = {wd: idx + 1 for idx, wd in gensim_dict.items()}  # 词索引字典 {词: 索引}，索引从1开始计数
            self._word2index['<unk>'] = self.UNK
            self._index2word = {idx: wd for wd, idx in self._word2index.items()}
            # reverse = lambda x:dict(zip(x, range(len(x))))  # 匿名函数输入是x，输出dict；eg：zip(x_list, (0,1,2,3))再转成dict
            # self._index2word = reverse(self._word2index)
            # word_vectors = {wd: wd2vec_model.wv[wd] for wd in self._word2index.keys()}  # 词向量 {词: 词向量}

            vocab_size = len(self._word2index)  # 语料词典大小(索引数字的个数)
            embedding_weights = np.zeros((vocab_size, embedding_size))  # vocab_size * EMBEDDING_SIZE的0矩阵
            for idx, wd in self._index2word.items():  # 从索引为1的词语开始，用词向量填充矩阵
                # embedding_weights[idx, :] = word_vectors[wd]
                if idx != self.UNK:
                    word_vector = wvmodel.wv[wd]
                    embedding_weights[idx] = word_vector
                    embedding_weights[self.UNK] += word_vector

            # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
            # embedding_weights[self.UNK, :] = np.random.uniform(-0.25, 0.25, config.embedding_size)
            # embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, config.embedding_size)
            embedding_weights[self.UNK] = embedding_weights[self.UNK] / vocab_size # 对oov进行初始化

            return embedding_weights

    # # 方法三
    # def get_embedding_weights(self, embed_path):
    #     # 保存每个词的词向量
    #     ch2vec_tab = {}
    #     vector_size = 0
    #     with open(embed_path, 'r', encoding='utf-8', errors='ignore') as fin: # 打开预训练词向量建立词向量表
    #         for line in fin:
    #             tokens, vector = line.split()
    #             vector_lst = []
    #             for each in vector:
    #                 vector_lst.append(each)
    #             # vector_size = len(tokens) - 1  # 第一个是表示词
    #             vector_size = len(vector)
    #             ch2vec_tab[tokens] = vector_lst
    #             # ch2vec_tab[tokens[0]].append(list(map(lambda x: float(x), tokens[1:])))  # map作用是把300维词向量都转成float型  ；这里得到的结果是一个word对应一个列表
    #     ch2vec_tab = dict(zip(token[0],list(map(lambda x: float(x), tokens[1:]))))  ###############
    #     self._char2idx = {ch: idx + 1 for idx, ch in enumerate(ch2vec_tab.keys())}  # 词索引字典 {词: 索引}，索引从1开始计数
    #     self._char2idx['<unk>'] = self._UNK
    #     self._idx2char = {idx: ch for ch, idx in self._char2idx.items()}
    #
    #     vocab_size = len(self._char2idx)  # 词典大小(索引数字的个数)
    #     embedding_weights = np.zeros((vocab_size, vector_size), dtype='float32')  # vocab_size * EMBEDDING_SIZE的0矩阵
    #     for idx, wd in self._idx2char.items():  # 从索引为1的词语开始，用词向量填充矩阵
    #         if idx != self._UNK:
    #             embedding_weights[idx] = ch2vec_tab[wd]
    #         # embedding_weights[self._UNK] += ch2vec_tab[wd]
    #
    #     # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
    #     # embedding_weights[self._UNK, :] = np.random.uniform(-0.25, 0.25, vector_size)
    #     embedding_weights[self._UNK] = np.random.uniform(-0.25, 0.25, vector_size)
    #     # embedding_weights[self._UNK] = embdding_weights[self._UNK] / vocab_size
    #     embedding_weights = embedding_weights / np.std(embedding_weights)  # 归一化 #########################
    #     return embedding_weights


    # 保持Vocab对象
    def save(self, save_vocab_path):
        with open(save_vocab_path, 'wb') as fw:
            pickle.dump(self, fw)

    # 获取情感词典one-hot向量
    # 注意力监督：序列中存在于情感词典中的词，对应位置为1
    def lexicon_vec(self, ws):
        if isinstance(ws, list):
            # True = 1   False = 0
            return [int(w in self._lexicon) for w in ws]
        else:
            return int(ws in self._lexicon) # bool判断返回int类型的1或0

    # 获取标签对应的索引
    def tag2index(self, tags):
        if isinstance(tags, list):
            return [self._tag2idx.get(tag, self.UNK_TAG) for tag in tags] # 标签默认值self.UNK_TAG是-1；索引值从0开始
        return self._tag2idx.get(tags, self.UNK_TAG)

    # 获取索引对应的标签值
    def index2tag(self, ids):
        if isinstance(ids, list):
            return [self._idx2tag.get(i) for i in ids] # 索引对应的标签值是从0开始
        return self._idx2tag.get(ids)

    # 获取词表中的词索引
    def word2index(self, ws):
        if isinstance(ws, list):
            return [self._word2index.get(w, self.UNK) for w in ws]  # get(w, self.UNK)中的self.UNK是默认值0
        return self._word2index.get(ws, self.UNK)

    # 获取词表中索引对应的词
    def index2word(self, ids):
        if isinstance(ids, list):
            return [self._index2word[i] for i in ids]
        else:
            return self._index2word[ids]

    # 获取语料中的词索引
    def corpus_wd2idx(self, ws):
        if isinstance(ws, list):
            return [self._corpus_wd2idx.get(w, self.UNK) for w in ws]  # 词索引是从1开始
        return self._corpus_wd2idx.get(ws, self.UNK)

    # 获取索引对应的语料中的词
    def corpus_idx2wd(self, ids):
        if isinstance(ids, list):
            return [self._corpus_idx2wd[i] for i in ids]
        else:
            return self._corpus_idx2wd[ids]

    # 获取语料词表的长度
    @property
    def corpus_vocab_size(self):
        return len(self._corpus_wd2idx)

    # 获取标签数（相当于分类的类别数）
    @property
    def tag_size(self):
        return len(self._tag2idx)

