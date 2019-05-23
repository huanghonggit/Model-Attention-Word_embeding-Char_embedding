import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import os
from module.char_encoder import CharEncoder
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
batch_first = False
LSTM输入: input, (h_0, c_0)
    - input (seq_len, batch, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.lstm.pack_padded_sequence(input, lengths, batch_first=False[source])
    - h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    - c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

LSTM输出 output, (h_n, c_n)
    - output (seq_len, batch, hidden_size * num_directions): 保存lstm最后一层的输出的Tensor。 如果输入是torch.nn.utils.lstm.PackedSequence，那么输出也是torch.nn.utils.lstm.PackedSequence。
    - h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的隐状态。
    - c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的细胞状态。


batch_first = True
LSTM输入: input, (h_0, c_0)
    - input (batch, seq_len, input_size): 包含输入序列特征的Tensor。也可以是packed variable ，详见 [pack_padded_sequence](#torch.nn.utils.lstm.pack_padded_sequence(input, lengths, batch_first=False[source])
    - h_0 (num_layers * num_directions, batch, hidden_size):保存着batch中每个元素的初始化隐状态的Tensor
    - c_0 (num_layers * num_directions, batch, hidden_size): 保存着batch中每个元素的初始化细胞状态的Tensor

LSTM输出 output, (h_n, c_n)
    - output (batch, seq_len, hidden_size * num_directions): 保存lstm最后一层的输出的Tensor。 如果输入是torch.nn.utils.lstm.PackedSequence，那么输出也是torch.nn.utils.lstm.PackedSequence。
    - h_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的隐状态。
    - c_n (num_layers * num_directions, batch, hidden_size): Tensor，保存着lstm最后一个时间步的细胞状态。
'''


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.correlation = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),       # nn.Tanh()
            nn.Linear(64, 1)
        )

    def forward(self, encoder_output):  # [batch_size, seq_len, hidden_size]
        a = self.correlation(encoder_output)  # [batch_size, seq_len, 1]
        weights = F.softmax(a.squeeze(-1), dim=1)  # 去掉a中指定的维数为1的维度 # [batch_size, seq_len]
        out = (encoder_output * weights.unsqueeze(-1)).sum(dim=1) # (64,100,128)*()  # [batch_size, hidden_size]
        return out, weights


class Attention_LSTM(nn.Module):
    def __init__(self, config, embedding_weights, char_weights):
        super(Attention_LSTM, self).__init__()

        # 先不加上语料这一层
        # embed_init = torch.zeros((vocab.corpus_vocab_size, embedding_dim), dtype=torch.float32)
        # self.corpus_embeddings = nn.Embedding(num_embeddings=vocab.corpus_vocab_size, embedding_dim=embedding_dim)
        # self.corpus_embeddings.weight.corpus.copy_(embed_init)
        # self.corpus_embeddings.weight.requires_grad = False # 加载的别人的预训练词向量不需要更新梯度

        # self.corpus_embeddings.weight = nn.Parameter(embed_init)
        self.config = config
        self.word_embedding_dim = embedding_weights.shape[1]
        self.char_embedding_dim = char_weights.shape[1]
        self.wd2vec_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))
        self.wd2vec_embeddings.weight.requires_grad = False  # [22667, 300]

        self.bidirectional = True
        self.nb_directions = 2 if self.bidirectional else 1
        self.lstm_dropout = self.config.drop_rate if self.config.nb_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=self.word_embedding_dim + config.char_hidden_size,  # 输入的特征维度(300+64)
                            hidden_size=self.config.hidden_size,  # 隐层状态的特征维度128
                            num_layers=self.config.nb_layers,  # LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果
                            dropout=self.lstm_dropout,  # 除了最后一层外，其它层的输出都会套上一个dropout层
                            bidirectional=self.bidirectional,  # 是否为双向LSTM
                            batch_first=True)  # [batch_size, seq, feature]
        self.char_encoder = CharEncoder(
            config=config,
            char_embedding_weights=char_weights
        )
        # self.self_attention = SelfAttention(self.nb_directions * self.config.hidden_size)
        self.self_attention = SelfAttention(self.config.hidden_size)
        self.dropout_embed = nn.Dropout(self.config.drop_embed_rate)
        self.dropout = nn.Dropout(self.config.drop_rate)
        # self.out = nn.Linear(self.nb_directions * self.config.hidden_size, self.config.nb_class)
        self.out = nn.Linear(self.config.hidden_size, self.config.nb_class)

    def init_hidden(self, batch_size):  # batch_size默认是64  这里的batch应该是在压紧边长序列之后的传入lstm的有效batch长度，而不是直接就是batch_size
        torch.cuda.manual_seed(3347)
        h_0 = torch.randn((self.config.nb_layers * self.nb_directions, batch_size, self.config.hidden_size))  # 根据中间隐藏层数量进行初始化(1*2, 64, 128)
        c_0 = torch.randn((self.config.nb_layers * self.nb_directions, batch_size, self.config.hidden_size))

        if self.config.use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        h_0 = Variable(h_0)
        c_0 = Variable(c_0)

        return h_0, c_0

    def forward(self, wd2vec_inputs, chars, seq_lens, config):  # (h0_state, c0_state)  这里是在train训练的时候拿到的数据
        # batch_size = inputs.shape[0] # inputs是一张有64句话的表

        init_hidden = self.init_hidden(wd2vec_inputs.shape[0])  # 先进行初始化，这里的batch_size应该是在压紧变长序列后的batch_长度;

        char_repre = self.char_encoder(chars) # char_weights
        wd2vec_inputs = wd2vec_inputs.to('cuda:0')

        wd_embed = self.wd2vec_embeddings(wd2vec_inputs) # wd_weights
        embed = torch.cat((wd_embed, char_repre), dim=2) # 把word_embedding + char_embedding合在一块  (64,301,300)+(64,301,64) = (64,301,364)
        # 语料层
        # corpus_embed = self.corpus_embeddings(inputs)
        # wd2vec_embed = self.wd2vec_embeddings(wd2vec_inputs)  # [64, 100, 300]
        # embed = corpus_embed + wd2vec_embed

        if self.training:  # 训练过程中采用Dropout，预测时False
            embed = self.dropout_embed(embed)
        # 使用pack_padded_sequence来确保LSTM模型不会处理用于填充的元素，可以理解成，把补0 的0去掉，对一个变长的序列进行压紧
        packed_embed = nn_utils.rnn.pack_padded_sequence(embed, seq_lens.cpu(), batch_first=True)  # .numpy()   batch_first=True  输出是   b, seq_len,size(64, 301, 364);返回值是一个tuple(压紧之后的序列，序列中的长度列表)
        # 保存着lstm最后一层的输出特征和最后一个时刻隐状态
        # h_0 = torch.randn((self.config.nb_layers * self.nb_directions, config.batch_size, self.config.hidden_size))
        # c_0 = torch.randn((self.config.nb_layers * self.nb_directions, config.batch_size, self.config.hidden_size))

        r_out, hidden = self.lstm(packed_embed, init_hidden)  # None 表示0初始化  , init_hidden    Variable(h_0.cuda()), Variable(c_0.cuda())
        r_out, _ = nn_utils.rnn.pad_packed_sequence(r_out, batch_first=True) # 可以理解成加上补0

        if self.bidirectional:
            r_out = r_out[:, :, :self.config.hidden_size] + r_out[:, :, self.config.hidden_size:]

        out, weights = self.self_attention(r_out)  # out:[64, 128]  # weights:[64, 100]

        if self.training:  # 训练过程中采用Dropout，预测时False
            out = self.dropout(out)
        out = self.out(out)

        return out, weights

