import argparse
from data import data
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# import matplotlib.pyplot as plt
import driver.HyperConfig as Config
from nets import Attention_LSTM as Model
from module import loss_function as LossFunc


# from sklearn.model_selection import train_test_split
# from torch.utils.corpus import DataLoader, TensorDataset
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def calc_acc(out, yb):
    pred = torch.argmax(F.softmax(out, dim=1), dim=1)
    acc = torch.eq(pred, yb).sum().item()
    return acc


def calc_loss(loss_func, out, yb, att_loss, weights, att_ids):
    loss_cls = loss_func(out, yb)
    loss_att = att_loss(weights, att_ids)
    loss = loss_cls + config.theta * loss_att
    return loss


# def draw(acc_lst, loss_lst):
# 	assert len(acc_lst) == len(loss_lst)
# 	nb_epochs = len(acc_lst)
# 	plt.subplot(211)
# 	plt.plot(list(range(nb_epochs)), loss_lst, c='r', label='loss')
# 	plt.legend(loc='best')
# 	plt.xlabel('epoch')
# 	plt.ylabel('loss')
# 	plt.subplot(212)
# 	plt.plot(list(range(nb_epochs)), acc_lst, c='b', label='acc')
# 	plt.legend(loc='best')
# 	plt.xlabel('epoch')
# 	plt.ylabel('acc')
# 	plt.tight_layout()
# 	plt.show()


# 测试集评估模型
def evaluate(test_data, model, charvocab, wordvocab, config, loss_func, att_loss):
    test_acc, test_loss = 0, 0
    numb_total = 0
    model.eval()
    for batch_data in data.get_batch(test_data, config.batch_size):  # 批训练  dev_data
        wd2vec_xb, yb, att_ids, char_vec, seqs_len = data.batch_data_variable(batch_data, wordvocab, charvocab)
        if config.use_cuda:
            # wd2vec_idxs = corpus_xb.cuda()
            char_vec = char_vec.cuda()
            wd2vec_xb = wd2vec_xb.cuda()
            yb = yb.cuda()
            att_ids = att_ids.cuda()
            seqs_len = seqs_len.cuda()
            char_vec = char_vec.cuda()

        out, weights = model(wd2vec_xb, char_vec, seqs_len, config)
        loss = calc_loss(loss_func, out, yb, att_loss, weights, att_ids)
        test_loss += loss.item()

        acc = calc_acc(out, yb)
        test_acc += acc

    # numb_total += np.sum(seqs_len)

    test_acc = float(test_acc) / len(test_data)
    print('test::epoch loss: %f, epoch acc : %f' % (test_loss, test_acc))


# 训练模型
def train(train_data, test_data, dev_data, charvocab, wordvocab, config):
    # loss_func = nn.NLLLoss()
    word_weights = wordvocab.get_word_embedding_weights(config.word2vec_path)
    char_weights = charvocab.get_char_embedding_weights(config.char2vec_path)  # , config.embedding_size

    model = Model.Attention_LSTM(config, word_weights, char_weights)

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)  # 可以改
    loss_func = nn.CrossEntropyLoss()  # 标签必须为0~n-1，而且必须为1维的
    att_loss = LossFunc.AttentionCrossEntropy()  # 监督注意力损失函数

    if config.use_cuda:
        torch.cuda.device(args.gpu)
        # torch.device('cuda:0')
        model = model.cuda()

    tocal_acc, total_loss = [], []
    for eps in range(config.epochs):
        model.train()  # 对每一轮都要进行train()
        print(' --Epoch %d' % (1 + eps))
        epoch_loss, epoch_acc = 0, 0
        # numb_total = 0
        t1 = time.time()
        for batch_data in data.get_batch(train_data, config.batch_size):  # 批训练
            # batch_data, seqs_len, _ = corpus.pad_batch(batch_data, config.max_len)
            # wd2vec_idxs, tags, att_ids, char_idxs, sorted_seq_lens
            wd2vec_xb, yb, att_ids, char_vec, seqs_len = data.batch_data_variable(batch_data, wordvocab, charvocab)
            if config.use_cuda:
                # wd2vec_idxs = corpus_xb.cuda()
                wd2vec_xb = wd2vec_xb.cuda()
                yb = yb.cuda()
                att_ids = att_ids.cuda()
                seqs_len = seqs_len.cuda()
                char_vec = char_vec.cuda()

            out, weights = model(wd2vec_xb, char_vec, seqs_len, config)

            # 重置模型梯度为0
            model.zero_grad()
            # yb = Variable(yb, required_grad=False)  # 张老师代码对正确标签进行了Variable

            # 计算损失
            loss = calc_loss(loss_func, out, yb, att_loss, weights, att_ids)
            epoch_loss += loss.item()

            # 计算准确率
            # if config.use_cuda:
            #     yb = yb.to('cuda:0')

            acc = calc_acc(out, yb)
            epoch_acc += acc

            # numb_total += seqs_len.sum().item()

            # 3.4 反向传播求梯度
            loss.backward()

            # 3.5 (用新的梯度值)更新模型参数
            optimizer.step()

        t2 = time.time()
        print('train_time：%.3f min' % ((t2 - t1) / 60))
        epoch_acc = float(epoch_acc) / len(train_data)
        print('train::epoch loss: %f, epoch acc: %f' % (epoch_loss, epoch_acc))
        total_loss.append(epoch_acc)
        tocal_acc.append(epoch_acc)

        with torch.no_grad():
            dev_total_acc, dev_total_loss = 0, 0
            numb_total = 0
            model.eval()
            for batch_data in data.get_batch(dev_data, config.batch_size):  # 批训练  dev_data
                wd2vec_xb, yb, att_ids, char_vec, seqs_len = data.batch_data_variable(batch_data, wordvocab, charvocab)
                if config.use_cuda:
                    # wd2vec_idxs = corpus_xb.cuda()
                    wd2vec_xb = wd2vec_xb.cuda()
                    yb = yb.cuda()
                    att_ids = att_ids.cuda()
                    seqs_len = seqs_len.cuda()
                    char_vec = char_vec.cuda()
                yb = yb.to('cuda:0')
                out, weights = model(wd2vec_xb, char_vec, seqs_len, config)
                loss = calc_loss(loss_func, out, yb, att_loss, weights, att_ids)
                dev_total_loss += loss.item()

                acc = calc_acc(out, yb)
                dev_total_acc += acc

            # numb_total += seqs_len.cpu().sum().item()

            dev_total_acc = float(dev_total_acc) / len(dev_data)
            print('dev::epoch loss: %f, epoch acc : %f' % (dev_total_loss, dev_total_acc))

    # draw(acc_lst, loss_lst)
    torch.save(model, config.save_model_path)

    evaluate(test_data, model, charvocab, wordvocab, config, loss_func, att_loss)  #####


if __name__ == '__main__':
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    gpu = torch.cuda.is_available()
    print("gpu", gpu)

    argparser = argparse.ArgumentParser()  # 实例化ArgumentParser类生成argparser解析对象
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=1, type=int, help='Use id of gpu, -1 if cpu.')
    args, extra_args = argparser.parse_known_args()  # parse_known_args()和parse_args()相比，在接受多余的命令行参数时不报错，返回的是一个tuple类型，包含两个元素，一个是命名空间一个是多余的命令列表
    torch.set_num_threads(args.thread)

    print('GPU可用：', torch.cuda.is_available())
    print('CuDNN：', torch.backends.cudnn.enabled)
    print('GPUs：', torch.cuda.device_count())
    print("GPU ID: ", args.gpu)

    config = Config.Config('driver/hyper_param.cfg')
    # 使用word_embedding 和char_embedding提升词向量的质量
    # char_vocab, word_vocab = create_vocab(config.train_data_path)

    train_data = data.load_data_instance(config.train_data_path)  # 以|||切分数据，拿到label,text,组装train_data的+inst
    test_data = data.load_data_instance(config.test_data_path)
    dev_data = data.load_data_instance(config.dev_data_path)

    charvocab = data.createCharVocab()  # 建立charvocab时不需要加载情感词   corpus_path=config.train_data_path
    charvocab.save(config.save_charvocab_path)

    wordvocab = data.createWordVocab(corpus_path=config.train_data_path,
                                     lexicon_path=config.lexicon_path)  # 用训练数据创建vocab对象
    wordvocab.save(config.save_wordvocab_path)
    train(train_data=train_data, test_data=test_data, dev_data=dev_data, charvocab=charvocab, wordvocab=wordvocab,
          config=config)

# config = Config('config/hyper_param.cfg')
# 	config.use_cuda = torch.cuda.is_available()  # cuda 的使用应该是根据这个判断
# 	if config.use_cuda:
# 		torch.cuda.set_device(1)

# 只保存模型参数
# torch.save(att_lstm.state_dict(), config.save_model_path)

# # 在每一轮结束使用开发集进行评测
# dev_acc = evaluate(test_data, att_lstm, vocab, config)
# if dev_acc > best_acc:
#     best_acc = dev_acc
#     last_step = steps
#     if args.save_best:
#         print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
#         save(model, args.save_dir, 'best', steps)
# else:
#     if steps - last_step >= args.early_stopping:
#         print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
#         raise KeyboardInterrupt
