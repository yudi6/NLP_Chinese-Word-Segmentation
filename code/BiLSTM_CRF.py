import torch
import torch.nn as nn

torch.manual_seed(1)
use_gpu = torch.cuda.is_available()
START_TAG = "<START>"
END_TAG = "<END>"
PAD_TAG = "<PAD>"


# 返回每行的最大值
def ArgMax(Arg):
    num, idx = torch.max(Arg, 1)
    return idx.item()


def LogSumExp(vector):
    max_score = vector[0, ArgMax(vector)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vector.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vector - max_score_broadcast)))


def LogSum(vector):
    return torch.log(torch.sum(torch.exp(vector), axis=0))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_rate, vocab_size, tag_to_ix, word_weight=None):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输出向量大小 也为字向量大小
            hidden_size=hidden_size // 2,  # 隐状态输出向量大小 双向则为1/2
            num_layers=num_layers,  # 层数
            bidirectional=True,  # 双向
            batch_first=True)  # 是否batch
        # self.word_embeds = nn.Embedding(vocab_size, input_size)             # 采用随机初始化的词向量 并做训练
        self.word_embeds = nn.Embedding.from_pretrained(word_weight)  # 采用训练的词向量
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden2tag = nn.Linear(hidden_size, self.tag_size)  # 线性层从隐状态向量到标签得分向量
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))  # CRF的转移矩阵表示从列序号对应标签转换到行序号对应标签
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 任意标签不能转移到start标签
        self.transitions.data[:, tag_to_ix[END_TAG]] = -10000  # end标签不能转移到任意标签
        self.hidden = self.HiddenInit()  # 隐藏层初始化

    def HiddenInit(self):
        return torch.randn(2, 1, self.hidden_size // 2), torch.randn((2, 1, self.hidden_size // 2))

    def ForwardAlg(self, feats):
        if use_gpu:
            init_alphas = torch.full([feats.shape[0], self.tag_size], -10000.).cuda()
        else:
            init_alphas = torch.full([feats.shape[0], self.tag_size], -10000.)
        # 开始标签的转换得分为0
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        # 输入的每个句子都进行前向算法
        forward_var_list = []
        forward_var_list.append(init_alphas)
        # 每个句子从句首开始迭代
        for feat_index in range(feats.shape[1]):
            # 迭代到某一词的logsumexp
            tag_score_now = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            feats_batch = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)
            # 新词的所有转移路径
            next_tag_score = tag_score_now + feats_batch + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(next_tag_score, dim=2))
        # 加上end标签的得分
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[END_TAG]].repeat([feats.shape[0], 1])
        # 每个句子进行logsumexp得到最终结果
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    # 单个句子经过lstm层加全连接层 输入为二位张量 变成三维张量
    def GetFeats(self, sentence):
        self.hidden = self.HiddenInit()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.squeeze()
        feats = self.hidden2tag(lstm_out)
        return feats

    # batch版 输入为三位张量
    def GetFeatsBatch(self, sentence):
        self.hidden = self.HiddenInit()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out)
        return feats

    # 给定序列tags的得分
    def Score(self, feats, tags):
        if use_gpu:
            score = torch.zeros(tags.shape[0]).cuda()
        else:
            score = torch.zeros(tags.shape[0])
        if use_gpu:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long().cuda(), tags], dim=1)
        else:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long(), tags], dim=1)
        for i in range(feats.shape[1]):
            # 第i个词得到的feats 二维张量
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[
                        range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[END_TAG], tags[:, -1]]
        return score

    # 维特比算法解码
    def Viterbi(self, feats):
        backpointers = []
        if use_gpu:
            init_vvars = torch.full((1, self.tag_size), -10000.).cuda()
        else:
            init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        # 三维的得分张量
        forward_var_list = []
        forward_var_list.append(init_vvars)
        # 从句子第一个词开始迭代
        for feat_index in range(feats.shape[0]):
            tag_score_now = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            tag_score_now = torch.squeeze(tag_score_now)
            next_tag_var = tag_score_now + self.transitions
            # 得到的最大得分
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)
            feat_batch = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + feat_batch
            forward_var_list.append(forward_var_new)
            # 标注得分的标签
            backpointers.append(bptrs_t.tolist())
        # 最终得分加入结束标志
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[END_TAG]]
        # 从后往前解码
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        # 根据tag_id进行解码
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        # 反转则是最终解码结果
        best_path.reverse()
        return path_score, best_path

    def LossFuction(self, sentences, tags):
        feats = self.GetFeatsBatch(sentences)
        forward_score = self.ForwardAlg(feats)
        gold_score = self.Score(feats, tags)
        # 所有输出句子的误差和作为结果
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):
        lstm_feats = self.GetFeats(sentence)
        score, tag_seq = self.Viterbi(lstm_feats)
        return score, tag_seq
