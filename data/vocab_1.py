import pickle
import gensim
import numpy as np
# from gensim.corpora import Dictionary


class CharVocab(object):
    # 法一
    # def __init__(self, word_counter, tag_counter, min_occur_count = 2):
    #     self._id2word = ['<pad>', '<bos>', '<eos>', '<unk>']
    #     self._wordid2freq = [10000, 10000, 10000, 10000] # 这里为什么是1w？
    #     self._id2tag = []
    #     for word, count in word_counter.most_common():  # most_common()返回最常见的元素及计数，顺序为   最常见到最少
    #         if count > min_occur_count:
    #             self._id2word.append(word) # 列表默认有id，只需要加进去词
    #             self._wordid2freq.append(count)   # 统计每个词出现次数
    #
    #     for tag, count in tag_counter.most_common():  # 这里只有2类？
    #         self._id2tag.append(tag)
    #
    #     reverse = lambda x: dict(zip(x, range(len(x))))  # 这里匿名函数输入是x，输出表达是dict ;eg: 把列表里面的每个词打包成zip；并转成dict格式；zip(word, (0,1,2,3))
    #     self._word2id = reverse(self._id2word)   # 使用reverse;id2word 转成 word2id
    #     if len(self._word2id) != len(self._id2word):
    #         print("serious bug: words dumplicated, please check!")
    #
    #     self._tag2id = reverse(self._id2tag)
    #     if len(self._tag2id) != len(self._id2tag):
    #         print("serious bug: POS tags dumplicated, please check!")
    #
    #     print("Vocab info: #words %d, #tags %d" % (self.vocab_size, self.tag_size))
    UNK = 0
    def __init__(self):  # 传进来的应该是一个char set,char_set
        super(CharVocab, self).__init__()
        self._idx2char = []
        self._char2idx = {}
        # self._UNK = 0  # 在这里定义时？？？？73行开始的for循环会执行到80行？？？


    # 取出预训练char向量赋值给权值矩阵
    def get_char_embedding_weights(self, embfile):
        embedding_dim = -1
        allwords = set()
        for special_word in ['<unk>']:  # 把'<pad>', '<bos>', '<eos>', '<unk>'特殊词如果不在 allwords则加入到set()里面
            if special_word not in allwords:
                allwords.add(special_word)
                self._idx2char.append(special_word)

        with open(embfile, encoding='utf-8') as f:  # 打开预训练词向量文件,统计char数量和维度
            for line in f.readlines():
                values = line.split()  # 以词向量的空格进行切分
                if len(values) > 10:
                    curword = values[0]  # values[0]是word
                    if curword not in allwords:
                        allwords.add(curword)                           # 把词存进set     这个allwords有什么用吗，预训练的词向量应该没有重复的把？
                        self._idx2char.append(curword)  # 对词存进列表进行编号 (extend:id)
                    embedding_dim = len(values) - 1
        char_num = len(self._idx2char)  # 拿到预训练词向量所以词数量
        print('Total chars: ' + str(char_num) + '\n')
        print('The dim of pretrained char embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._char2idx = reverse(self._idx2char)  # 建立预训练词向量(id:extend)字典

        if len(self._char2idx) != len(self._idx2char):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._char2idx.get('<unk>')  # oov:Out of Vocabulary
        # if self.UNK != oov_id:  # 判断oov是否对应
        #     print("serious bug: oov word id is not correct, please check!")

        char_embeddings = np.zeros((char_num, embedding_dim))  # 创建一个全为0的预训练词表
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    index = self._char2idx.get(values[0])  # 拿到词的索引，同时也是在char_embedding表里面的索引
                    vector = np.array(values[1:], dtype='float64') # 取出value[1:]部分，转为np.array 赋值给vector
                    char_embeddings[index] = vector  # 整个矩阵全是权值列表，embeddings[index] = vector  表示index行的vector
                    char_embeddings[self.UNK] += vector  # 把每一行权值对应相加给self.UNK
        char_embeddings[self.UNK] = char_embeddings[self.UNK] / char_num  # 把oov词全赋值为 均值
        char_embeddings = char_embeddings / np.std(char_embeddings)
        return char_embeddings

    def save(self, save_char_path):
        with open(save_char_path, 'wb') as fw:
            pickle.dump(self, fw)

    def char2idx(self, chars):  # 可 爱
        if len(chars) > 1:
            return [self._char2idx.get(c, self.UNK) for c in chars]
        else:
            return [self._char2idx.get(chars, self.UNK)]  # 传回去的是一个列表装的数值

    def idx2char(self, xs):
        if isinstance(xs, list):
            return [self._idx2char[x] for x in xs]
        return self._idx2char[xs]

    @property
    def vocab_size(self):
        return len(self._idx2char)

    @property
    def charvocab_size(self):
        return len(self._idx2char)


    # 法二
    # def __init__(self):  # 传进来的应该是一个char set,char_set
    #     super(CharVocab, self).__init__()
    #     self.UNK = 0
    #     self._char2idx = None
    #     self._idx2char = None
    #
    # # self._char2idx = {char: idx+1 for idx, char in enumerate(char_set)}
    # # self._char2idx['un'] = self.UNK
    # # self._idx2char = {idx: char for char, idx in self._char2idx.items()}
    #
    # def get_embedding_weights(self, embed_path):
    #     # 保存每个词的词向量
    #     ch2vec_tab = {}
    #     vector_size = 0
    #     with open(embed_path, 'r', encoding='utf-8', errors='ignore') as fin:  # 拿到外边已经训练好的字符向量
    #         for line in fin:
    #             tokens = line.split()
    #             vector_size = len(tokens) - 1
    #             ch2vec_tab[tokens[0]] = list(map(lambda x: float(x), tokens[1:]))
    #
    #     self._char2idx = {ch: idx for idx, ch in enumerate(ch2vec_tab.keys())}  # 词索引字典 {词: 索引}，索引从1开始计数
    #     self._char2idx['<unk>'] = self.UNK
    #     self._idx2char = {idx: ch for ch, idx in self._char2idx.items()}
    #
    #     vocab_size = len(self._char2idx)  # 词典大小(索引数字的个数)
    #     embedding_weights = np.zeros((vocab_size, vector_size), dtype='float32')  # vocab_size * EMBEDDING_SIZE的0矩阵
    #     for idx, wd in self._idx2char.items():  # 从索引为1的词语开始，用词向量填充矩阵
    #         if idx != self.UNK:
    #             embedding_weights[idx] = ch2vec_tab[wd]
    #             embedding_weights[self.UNK] += ch2vec_tab[wd]
    #
    #     # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
    #     # embedding_weights[self.UNK, :] = np.random.uniform(-0.25, 0.25, vector_size)
    #     # embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, vector_size)
    #     embedding_weights[self.UNK] = embedding_weights[self.UNK] / vocab_size
    #     embedding_weights = embedding_weights / np.std(embedding_weights)  # 归一化
    #     return embedding_weights

    # def save(self, save_char_path):
    #     with open(save_char_path, 'wb') as fw:
    #         pickle.dump(self, fw)
    #
    # def char2idx(self, chars):  # 可 爱
    #     if len(chars) > 1:
    #         return [self._char2idx.get(c, self.UNK) for c in chars]
    #     else:
    #         return self._char2idx.get(chars, self.UNK)
    #
    # def idx2char(self, idxs):
    #     if isinstance(idxs, list):
    #         return [self._idx2char[i] for i in idxs]
    #     else:
    #         return self._idx2char[idxs]
    #
    # @property
    # def vocab_size(self):
    #     return len(self._char2idx)

# 词表
class WordVocab(object):
    UNK, UNK_TAG = 0, -1  # 文本默认值和默认标签值

    def __init__(self, wds_counter, tags_counter, lexicon_path=None):
        self.min_count = 5
        # self._word2index = {}
        # self._index2word = {}
        self._lexicon = set()

        self._wd2freq = {wd: count for wd, count in wds_counter.items() if
                         count > self.min_count}  # 输出是词典{word：count}，去掉出现次数小于2 的词

        self._corpus_wd2idx = {wd: idx + 1 for idx, wd in
                               enumerate(self._wd2freq.keys())}  # enumerate idx是从0开始   从word：1开始
        self._corpus_wd2idx['<unk>'] = self.UNK  # 针对oov一对{k:v}
        self._corpus_idx2wd = {idx: wd for wd, idx in self._corpus_wd2idx.items()}  # {idx : word}

        self._tag2idx = {tag: idx for idx, tag in
                         enumerate(tags_counter.keys())}  # tags_counter里面是tag（0,1,2）和对应的count； tag:0开始
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

    # 获得embedding权重向量和词索引表   改成手动建立dict
    def get_word_embedding_weights(self, embfile):  # 词向量路径和词向量的维度
        embedding_dim = -1
        self._idx2word = []
        allwords = set()
        for special_word in ['unk']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._idx2word.append(special_word)

        with open(embfile, encoding='utf-8') as f:  # 第一个打开预训练词向量文件，目的是为了统计里面的次数，拿到词的维度，建立词total_set
            for line in f.readlines():
                vals = line.split()  # 以词后面每个空格进行切分;
                if len(vals) > 10:  # 这里为啥是大于10，作用可以把比如第一行 标签名或者统计字符 不算入计数
                    curword = vals[0]
                    if curword not in allwords:
                        allwords.add(curword)  # set()添加词用add,list用append
                        self._idx2word.append(curword)  # 把词存进列表进行编号 从(0 : x)开始
                    embedding_dim = len(vals) - 1  # 词向量的维度
        word_num = len(self._idx2word)  # 拿到所以embed里面词向量个数
        print('embedfile total words:' + str(word_num) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2idx = reverse(self._idx2word)  # eg：[a,b,c]; out:dict{a:0, b:1, c:2}   (x : 0)

        if len(self._word2idx) != len(self._idx2word):
            print("serious bug: words dumplicated, please check!")

        # oov_id = self._word2idx['unk'].get('unk')
        # if self.UNK != oov_id:
        #     print("serious bug: oov word id is not correct, please check!")

        word_embedding = np.zeros((word_num, embedding_dim))  # word_num是预训练词表的词个数
        with open(embfile, encoding='utf-8') as f:  # 第二次打开预训练文件是为了读取词向量
            for line in f.readlines():
                vals = line.split()
                if len(vals) > 10:
                    idx = self._word2idx.get(vals[0])  # 拿到词向量文件的第一个词 ，从_word2idx里边拿到词对应的idx
                    vector = np.array(vals[1:], dtype='float64')  # 拿到每个词的词向量  float64
                    word_embedding[idx] = vector  # 直接映射在表中，word_embedding表的第idx行就是idx这个词的词向量
                    word_embedding[self.UNK] += vector
        word_embedding[self.UNK] = word_embedding[self.UNK] / word_num
        word_embedding = word_embedding / np.std(word_embedding)  # 这里是对矩阵里面每个值除以标准差
        return word_embedding

        # # 直接使用预训练词向量
        # wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
        # if wvmodel is not None:
        #     gensim_dict = Dictionary()  # {索引: 词}
        #     # 实现词袋模型
        #     gensim_dict.doc2bow(wvmodel.wv.vocab.keys(), allow_update=False)  # (token_id, token_count);当使用预训练词向量时，不需要更新词向量
        #     self._word2index = {wd: idx + 1 for idx, wd in gensim_dict.items()}  # 词索引字典 {词: 索引}，索引从1开始计数
        #     self._word2index['<unk>'] = self.UNK
        #     self._index2word = {idx: wd for wd, idx in self._word2index.items()}
        #     # reverse = lambda x:dict(zip(x, range(len(x))))  # 匿名函数输入是x，输出dict；eg：zip(x_list, (0,1,2,3))再转成dict
        #     # self._index2word = reverse(self._word2index)
        #     # word_vectors = {wd: wd2vec_model.wv[wd] for wd in self._word2index.keys()}  # 词向量 {词: 词向量}
        #
        #     vocab_size = len(self._word2index)  # 语料词典大小(索引数字的个数)
        #     embedding_weights = np.zeros((vocab_size, embedding_size))  # vocab_size * EMBEDDING_SIZE的0矩阵
        #     for idx, wd in self._index2word.items():  # 从索引为1的词语开始，用词向量填充矩阵
        #         # embedding_weights[idx, :] = word_vectors[wd]
        #         if idx != self.UNK:
        #             word_vector = wvmodel.wv[wd]
        #             embedding_weights[idx] = word_vector
        #             embedding_weights[self.UNK] += word_vector
        #
        #     # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
        #     # embedding_weights[self.UNK, :] = np.random.uniform(-0.25, 0.25, config.embedding_size)
        #     # embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, config.embedding_size)
        #     embedding_weights[self.UNK] = embedding_weights[self.UNK] / vocab_size # 对oov进行初始化
        #
        #     return embedding_weights

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
    #     self._char2idx['<unk>'] = self.UNK
    #     self._idx2char = {idx: ch for ch, idx in self._char2idx.items()}
    #
    #     vocab_size = len(self._char2idx)  # 词典大小(索引数字的个数)
    #     embedding_weights = np.zeros((vocab_size, vector_size), dtype='float32')  # vocab_size * EMBEDDING_SIZE的0矩阵
    #     for idx, wd in self._idx2char.items():  # 从索引为1的词语开始，用词向量填充矩阵
    #         if idx != self.UNK:
    #             embedding_weights[idx] = ch2vec_tab[wd]
    #         # embedding_weights[self.UNK] += ch2vec_tab[wd]
    #
    #     # 对于OOV的词赋值为0 或 随机初始化 或 赋其他词向量的均值
    #     # embedding_weights[self.UNK, :] = np.random.uniform(-0.25, 0.25, vector_size)
    #     embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, vector_size)
    #     # embedding_weights[self.UNK] = embdding_weights[self.UNK] / vocab_size
    #     embedding_weights = embedding_weights / np.std(embedding_weights)  # 归一化 #########################
    #     return embedding_weights

    # 保持Vocab对象
    def save(self, save_wordvocab_path):
        with open(save_wordvocab_path, 'wb') as fw:
            pickle.dump(self, fw)

    # 获取情感词典one-hot向量
    # 注意力监督：序列中存在于情感词典中的词，对应位置为1
    def lexicon_vec(self, ws):
        if isinstance(ws, list):
            # True = 1   False = 0
            return [int(w in self._lexicon) for w in ws]
        else:
            return int(ws in self._lexicon)  # bool判断返回int类型的1或0

    # 获取标签对应的索引
    def tag2index(self, tags):
        if isinstance(tags, list):
            return [self._tag2idx.get(tag, self.UNK_TAG) for tag in tags]  # 标签默认值self.UNK_TAG是-1；索引值从0开始
        return self._tag2idx.get(tags, self.UNK_TAG)

    # 获取索引对应的标签值
    def index2tag(self, ids):
        if isinstance(ids, list):
            return [self._idx2tag.get(i) for i in ids]  # 索引对应的标签值是从0开始
        return self._idx2tag.get(ids)

    # # 获取词表中的词索引
    # def word2index(self, ws):
    #     if isinstance(ws, list):
    #         return [self._word2idx.get(w, self.UNK) for w in ws]  # get(w, self.UNK)中的self.UNK是默认值0
    #     return self._word2idx.get(ws, self.UNK)
    #
    # # 获取词表中索引对应的词
    # def index2word(self, ids):
    #     if isinstance(ids, list):
    #         return [self._idx2word[i] for i in ids]
    #     else:
    #         return self._idx2word[ids]

    # 获取词表中的词索引
    def word2index(self, ws):  # self._word2idx是dict
        if isinstance(ws, list):
            return [self._word2idx.get(w, self.UNK) for w in ws]  # get(w, self.UNK)中的self.UNK是默认值0，拿到这个词对应的idx，也对应word_embedding 的第idx行的词向量
        return self._word2idx.get(ws, self.UNK)

    # 获取词表中索引对应的词
    def index2word(self, ids):
        if isinstance(ids, list):
            return [self._idx2word[i] for i in ids]
        return self._idx2word[ids]

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
