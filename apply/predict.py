import sys
sys.path.append(["../../", "../", "./"])
import torch
import torch.nn.functional as F
from data import data
import driver.HyperConfig as Config
import pickle

torch.manual_seed(3347)


def highlight_word(wd, **color):
    return '<span style="background-color: rgb(%d, %d, %d)">%s</span>' % (color['r'], color['g'], color['b'], wd)


def highlight_seq(wd_seq, idxs, weights, pred_no):
    marked_seq = []
    for i, wd in enumerate(wd_seq):
        if i in idxs:
            w = int(255 * (1 - weights[i]))
            color = {}
            if pred_no == 0:
                color = {'r': w, 'g': w, 'b': 255}
            elif pred_no == 1:
                color = {'r': w, 'g': 255, 'b': w}
            elif pred_no == 2:
                color = {'r': 255, 'g': w, 'b': w}

            wd = highlight_word(wd, **color)

        marked_seq.append(wd)

    return ''.join(marked_seq)


def load_vocab(load_vocab_path):
    with open(load_vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    return vocab


def predict(pred_data, vocab, config):
    # classifier = Attention_LSTM_SA.Attention_LSTM_SA()
    # classifier.load_state_dict(torch.load(config.load_model_path))  # 加载模型参数

    # 加载模型
    classifier = torch.load(config.load_model_path)

    classifier.eval()  # self.training = False

    insts = data.preprocess_data(pred_data)  # 单词序列

    corpus_idxs, wd2vec_idxs, _, _, seq_lens, origin_indices = data.batch_data_variable(insts, vocab, config)

    out, weights = classifier(corpus_idxs, wd2vec_idxs, seq_lens)
    print('out:{}'.format(out))
    print('weights:{}'.format(weights))
    emo_values = torch.sum(out, 0, False) # 把一段的每一句话输出对应的（0,1,2）值加起来，结果是总的（好，中，差）
    emo_values = F.softmax(emo_values, dim=-1) # 对一行进行softmax
    print("emo_values:", emo_values)

    pred = torch.argmax(F.softmax(out, dim=1), dim=1) # 拿到最大的列号
    # print('pred:', pred)
    weights = weights[origin_indices]
    pred = pred[origin_indices]
    col=[]
    for each in pred:
        col.append(each.item())
    topn = 5
    results = []
    # top_weights, top_idxs = torch.topk(weights, topn, dim=1)
    # for ws, idxs, wd_seq, pred_no in zip(top_weights, top_idxs, wd_seqs, pred.numpy()):
    #     hs = highlight_seq(wd_seq, idxs.corpus.numpy(), ws.corpus.numpy(), pred_no)
    #     print(hs)
    #     results.append(hs)

    for inst, ws, pred_no in zip(insts, weights, pred):
        _, top_idxs = torch.topk(ws, topn)
        hs = highlight_seq(inst.words, top_idxs.cpu(), ws.data.cpu(), pred_no) # .numpy()
        results.append(hs)

    return results, emo_values


# pred_1 = F.softmax(out, dim=1) # 对每行进行softmax
# emo_val = []
#     for each in pred_1:
#         crow_each = []
#         for one in each:
#             one = one.item()
#             crow_each.append(round(one, 4))
#         emo_val.append(crow_each)
#     print('emo_val:', emo_val)


if __name__ == '__main__':

    X = [
        '鞋子很快就收到了，质量过关，价格不贵，性价比 很高，鞋子穿得也比较舒服，尺码也标准就按平时 自己穿得码数买就行了！质量很好 版型也很好 码子很标准 穿上很有档次 卖家服务超级好 很满意的一次网上购物。',
        '发过来的鞋子跟图片不是同一款 没有图片上的好看 鞋子的鞋面跟鞋带都不一样 只有鞋底一样 太坑了 并且物流不是一般的慢',
        '鞋子感觉一般吧，穿上不是特别舒服，这个价钱中规中矩吧。',
        '实话这手机个人感觉很没档次，手感就感觉塑料一样的模型机一般，充电特慢，电池不够用，机后头轻敲都是响声，不知道是我运气不好还是，有时偶尔有点卡，玩游戏直接不行，网络延时得吓人，声音基本和动作不匹配，这个价格的话我介意各位买其他机器，这是我用的华为手机最差劲的一款，最后补充一点，就是连张普通膜都不舍得',
        '快递方面没得说很好，但是手机就算了，我用这是第6个华为了绝对的忠实粉没想到这次让我这么失望，我很少评价但是这次实在受不了，就下载了2个AP竟然卡了卡的比我用了快3年的荣耀6都卡好家伙我真无语了价格是很实惠但是也不至于卡到这个地步吧我很相信华为但这次真的很生气郁闷'
    ]

    config = Config.Config('config/hyper_param.cfg')
    vocab = data.createVocab(corpus_path=config.train_data_path, lexicon_path=config.lexicon_path)
    vocab.save(config.save_vocab_path)
    # vocab = load_vocab(config.load_vocab_path)
    predict(X, vocab, config)
