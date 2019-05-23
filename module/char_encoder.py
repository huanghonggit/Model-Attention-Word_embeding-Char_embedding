import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEncoder(nn.Module):
    def __init__(self, config, char_embedding_weights):
        super(CharEncoder, self).__init__()

        self.char_hidden_size = config.char_hidden_size
        self.vocab_size, self.char_embedding_size = char_embedding_weights.shape
        # self.char_embedding = nn.Embedding.from_pretrained(torch.from_numpy(char_embedding_weights))
        self.char_embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                           embedding_dim=self.char_embedding_size)
        # nn.init.uniform_(self.char_embedding.parameters(), -0.32, 0.32) # 从均匀分布U(-0.32, 0.32)中生成值，填充输入的张量或变量 #######################
        # self.char_embedding.weight.corpus.copy_(xx)
        # nn.init.uniform_(self.char_embedding.weight.corpus, -0.32, 0.32)
        # nn.init.uniform_(self.char_embedding.bias, 0, 0)

        self.win_sizes = [int(i) for i in config.char_window_size if i != ' ']#list(config.char_window_size)  [2, 3, 4]  # 为什么要加这个窗口大小,作用？
        self.padding = 1

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.char_embedding_size,  # 每个char的维度80
                          out_channels=self.char_hidden_size,  # charembedding层卷积隐藏层的数量64
                          padding=self.padding,  # 输入的每一条边补充0的层数
                          kernel_size=int(w)),
                nn.ReLU(),
            )

            for w in self.win_sizes ######################
        ])

        # self.win_sizes = [1, 2, 3, 4]
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=self.char_embedding_size,
        #             out_channels=config.char_hidden_size,
        #             padding=self.padding,
        #             kernel_size=w
        #         ),
        #         nn.ReLU(),
        #         nn.MaxPool1d(kernel_size=max_len + 2*self.padding - w + 1)
        #     ) for w in self.win_sizes
        # ])

        self.linear = nn.Linear(len(self.win_sizes) * self.char_hidden_size, self.char_hidden_size) # 全连接层为啥输入是len(self.win_sizes) * self.char_hidden_size,是窗口数相乘，怎么理解
        self.dropout = nn.Dropout(config.drop_rate)
        self.dropout_embed = nn.Dropout(config.drop_embed_rate)

    def conv_and_pool(self, x, conv, i):  # 输入的x维度(19264,80,4)
        # k = F.relu(x)  # 为什么卷积之前先进行relu()
        conv_out = conv(x) # 对窗口2,3,4分别进行卷积；k=2时，(19264,80,4)-->(19264,64,5)  64是卷积输出到隐藏层64， (4-2+2*1)/1 +1 =5;   k=3时，(19264,80,4)-->(19264,64,4)  64是卷积输出到隐藏层64， (4-3+2*1)/1 +1 =4; k=4时，(19264,80,4)-->(19264,64,3)   (4-4+2*1)/1 +1 =3
        conv_out = F.relu(conv_out)
        if i==1:
            out = F.max_pool1d(conv_out, conv_out.size(2))  # 经过池化后(19264,64,4)，(19264,64,5)，(19264,64,3)都-->(19264,64,1)
        else:
            out = F.avg_pool1d(conv_out, conv_out.size(2))

        return out

    # 根据输入长度计算卷积输出长度
    # stride=1时，一维卷积层输出大小(宽度) = （序列大小 + 2*pad - 窗口大小）/stride + 1
    # def conv_out_size(self, L):
    #     # stride = 1
    #     return (L + 2*self.padding - self.win_size) + 1

    def forward(self, chars):  # (batch_size, max_seq_len, max_wd_len)
        batch_size, max_seq_len, max_wd_len = chars.size()

        chars = chars.reshape((-1, max_wd_len))  # 把前面两维变一维(batch_size * max_seq_len, max_wd_len)
        chars = chars.to('cuda:0')

        embed_x = self.char_embedding(chars)  # (batch_size * max_seq_len, max_wd_len, char_embedding_size)

        # batch_size * max_len * embedding_size ->batch_size * embedding_size * max_len
        embed_x = embed_x.permute(0, 2, 1)  # (batch_size * max_seq_len, char_embedding_size, max_wd_len),需要这个操作的原因是，卷积是在最后的维度上面扫过的，我这里需要是在4这里扫过。这里卷积为啥是对4，而不是对80？

        if self.training:
            embed_x = self.dropout_embed(embed_x)

        # (batch_size * max_seq_len, char_hidden_size, conv_out) ->
        # (batch_size * max_seq_len, char_hidden_size, 1)
        # 2,3,4三次卷积结果对应使用最大，最小，平均池化
        # out = [self.conv_and_pool(embed_x, conv) for conv in self.convs] # 这里out存储三次卷积后最大池化的结果，即三个列表 （19264,64,1）(19264,64,1)(19264,64,1)；卷积用在nlp就是抽取连续char中的特征，可以理解最后拿到的是深度语义特征
        out = [self.conv_and_pool(embed_x, conv, i+1) for i, conv in enumerate(self.convs)]
        conv_out = torch.cat(tuple(out), dim=1)  # 对应第二个维度拼接起来，如 5*2*1,5*3*1的拼接变成5*5*1

        # 使用掩码处理padding过的不定长序列卷积结果
        # 1、根据实际序列长度计算卷积输出的长度
        # 2、在step 1的基础上生成mask
        # 3、mask乘以卷积输出的结果
        # conv_lens = self.conv_out_size(wd_lens.flatten())  # (batch_size, max_seq_len) -> (batch_size*max_seq_len, )
        # mask = torch.zeros_like(conv_out, device=chars.device)
        # for i, conv_len in enumerate(conv_lens):
        #     mask[i, :, :conv_len].fill_(1)
        # conv_out = conv_out * mask

        conv_out = conv_out.squeeze()  # 去掉维度为1的值去掉（19264,192,1）-->(19264,192)

        if self.training:
            conv_out = self.dropout(conv_out)

        out = self.linear(conv_out)

        out = out.reshape(batch_size, max_seq_len, -1) # 19264,64 --> 64,301,64

        return out
