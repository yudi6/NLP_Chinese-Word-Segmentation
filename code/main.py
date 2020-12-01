from tqdm import tqdm
from BiLSTM_CRF import *
from MyDataSet import *
import torch.optim as optim
import re
import csv
import gensim
import os
import random

tag_to_ix = {PAD_TAG: 0, "B": 1, "M": 2, "E": 3, "S": 4, START_TAG: 5, END_TAG: 6}

def Divide(data, partition):
    sample_number = int(len(data) * partition)
    random.shuffle(data)
    train_data = data[:sample_number]
    vali_data = data[sample_number:]
    return train_data, vali_data


def Tag2Word(sentence, tag, tag_to_ix, transform=False):
    if transform:
        ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        tag_true = [ix_to_tag[i] for i in tag]
    else:
        tag_true = tag
    word_list = []
    word = str()
    for i in range(len(tag_true)):
        if tag_true[i] == 'B':
            word = sentence[i]
        elif tag_true[i] == 'S':
            word_list.append(sentence[i])
        elif tag_true[i] == 'M':
            word += sentence[i]
        elif tag_true[i] == 'E':
            word += sentence[i]
            word_list.append(word)
        elif tag_true[i] == '<PAD>':
            break
    if word not in word_list:
        word_list.append(word)
    return word_list


# 使用Word2Vec训练字向量模型
def TrainWord2Vec(sentences, input_size, word_to_ix, load_word_model='Word2Vec'):
    weight = torch.zeros(len(word_to_ix), input_size)
    if load_word_model == 'baidu':
        Word2Vec_model = gensim.models.KeyedVectors.load('word_vector/baidubaike_300')
        Word2Vec_model_temp = gensim.models.Word2Vec(sentences, size=input_size, min_count=1)
        for i in range(len(Word2Vec_model_temp.wv.index2word)):
            word = Word2Vec_model_temp.wv.index2word[i]
            index = word_to_ix[word]
            if word in Word2Vec_model:
                weight[index, :] = torch.from_numpy(Word2Vec_model.wv.get_vector(word))
            else:
                weight[index, :] = torch.randn(input_size)
        weight[0, :] = torch.zeros(input_size)
    elif load_word_model == 'merge':
        Word2Vec_model = gensim.models.KeyedVectors.load('word_vector/merge_300')
        Word2Vec_model_temp = gensim.models.Word2Vec(sentences, size=input_size, min_count=1)
        for i in range(len(Word2Vec_model_temp.wv.index2word)):
            word = Word2Vec_model_temp.wv.index2word[i]
            index = word_to_ix[word]
            if word in Word2Vec_model:
                weight[index, :] = torch.from_numpy(Word2Vec_model.wv.get_vector(word))
            else:
                weight[index, :] = torch.randn(input_size)
        weight[0, :] = torch.zeros(input_size)
    # 使用原有的Word2Vec训练得到的向量
    elif load_word_model == 'Word2Vec':
        if os.path.exists('Word2Vec_300'):
            Word2Vec_model = gensim.models.Word2Vec.load('word_vector/Word2Vec_300')
        else:
            Word2Vec_model = gensim.models.Word2Vec(sentences, size=input_size, min_count=1)
            Word2Vec_model.save('Word2Vec_300')
        for i in range(len(Word2Vec_model.wv.index2word)):
            index = word_to_ix[Word2Vec_model.wv.index2word[i]]
            weight[index, :] = torch.from_numpy(Word2Vec_model.wv.get_vector(Word2Vec_model.wv.index2word[i]))
    return weight


def TruncAndPad(sentence_set, sentence_tag, max_len):
    # 句子长度大于设定长度则截断
    if (len(sentence_set) >= max_len):
        return sentence_set[0: max_len], sentence_tag[0:max_len]
    # 句子长度小于设定长度则在句子后增加“<PAD>”
    else:
        sentence_set.extend(["<PAD>"] * (max_len - len(sentence_set)))
        sentence_tag.extend(["<PAD>"] * (max_len - len(sentence_tag)))
    return sentence_set, sentence_tag


def GetSentenceTag(sentence_set, word_set):
    sentence_tag = ['S' for i in range(len(sentence_set))]
    index = 0
    for term in word_set:
        # 将每个分词拆分成字
        term_split = [one for one in term]
        word_length = len(term_split)
        # 只有一个字则标注为‘S’
        if (word_length == 1):
            sentence_tag[index] = 'S'
        # 多个字则按‘BME’顺序标注
        else:
            sentence_tag[index] = 'B'
            index += 1
            word_length -= 1
            while (word_length > 1):
                sentence_tag[index] = 'M'
                word_length -= 1
                index += 1
            sentence_tag[index] = 'E'
        index += 1
    return sentence_tag


def LoadData(file, load_from_csv=False, max_len=200):
    # 返回的数据元组列表
    data = []
    sentence_tag_list = []
    sentence_split_list = []
    # 将训练数据保存在csv文件中
    if load_from_csv:
        with open('sentence_tag_list.csv', 'r', newline='', encoding='utf-8') as fp:
            sentence_tag_list = [i for i in csv.reader(fp)]
        with open('sentence_split_list.csv', 'r', newline='', encoding='utf-8') as fp:
            sentence_split_list = [i for i in csv.reader(fp)]
    else:
        with open(file, 'r', encoding='utf-8') as fp:
            sentences = fp.readlines()
        for i in range(len(sentences)):
            remove_chars = '[ \\n]+'
            # 获得字列表
            sentence_split = [one for one in re.sub(remove_chars, "", sentences[i])]
            sentence_split_list.append(sentence_split)
            # 获得词列表
            remove_chars = '[\\n]+'
            word_set = re.sub(remove_chars, "", sentences[i]).split()
            # 根据字列表与词列表得到该句子的标签序列
            sentence_tag = GetSentenceTag(sentence_split, word_set)
            sentence_tag_list.append(sentence_tag)
        # 保存读取的数据 方便下次训练
        with open('sentence_tag_list.csv', 'w', newline='', encoding='utf-8') as fp:
            write = csv.writer(fp)
            write.writerows(sentence_tag_list)
        with open('sentence_split_list.csv', 'w', newline='', encoding='utf-8') as fp:
            write = csv.writer(fp)
            write.writerows(sentence_split_list)
    # 按元组形式保存训练数据
    for i in range(len(sentence_split_list)):
        data.append(TruncAndPad(sentence_split_list[i], sentence_tag_list[i], max_len))
    return data


def LoadTestData(file):
    test_data = []
    with open(file, 'r', encoding='utf-8') as fp:
        sentences = fp.readlines()
    for i in range(len(sentences)):
        # remove_chars = '[·’!"\#$%&\'()＃*+,-./:;<=>?\@，：?￥★、…．＞【】［］“”‘’\[\\]^_`{|}~ \\n]+'
        remove_chars = '[ \\n]+'
        sentence_split = [one for one in re.sub(remove_chars, "", sentences[i])]
        remove_chars = '[\\n]+'
        word_set = re.sub(remove_chars, "", sentences[i]).split()
        sentence_tag = GetSentenceTag(sentence_split, word_set)
        if len(sentence_split) == 1:
            test_data.append(TruncAndPad(sentence_split, sentence_tag, 2))
        else:
            test_data.append((sentence_split, sentence_tag))
    return test_data


def GetF1(model_result, true_result):
    index = 0
    model_result_index = []
    for word in model_result:
        model_result_index.append((index, index + len(word) - 1))
        index = index + len(word)
    index = 0
    true_result_index = []
    for word in true_result:
        true_result_index.append((index, index + len(word) - 1))
        index = index + len(word)
    count = 0
    for i in model_result_index:
        if i in true_result_index:
            count += 1
    precision = count / (len(model_result) + +0.000000001)
    recall = count / (len(true_result) + 0.00000000001)
    return (2 * precision * recall) / (precision + recall + 0.00000000001)


if __name__ == '__main__':
    INPUT_SIZE = 300  # 词向量大小
    HIDDEN_SIZE = 200  # 输出特征向量大小
    NUM_LAYERS = 2  # BiLSTM层数
    DROP_RATE = 0.5  # drop out rate
    EPOCH = 50   # 迭代次数
    LR = 0.0001  # 学习率
    BATCH_SIZE = 256    #
    MAX_LEN = 200   # 句子最大长度
    data = LoadData(r'msrseg/msr_training.utf8', False, MAX_LEN)
    test_data = LoadTestData(r'msrseg/msr_test_gold.utf8')
    word_to_ix = {"<PAD>": 0}
    all_sentence = []
    for sentence, tags in data:
        all_sentence.append(sentence)
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    for sentence, tags in test_data:
        all_sentence.append(sentence)
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    train_data, vali_data = Divide(data, 1)
    # 字向量文件
    word_weight = TrainWord2Vec(all_sentence, INPUT_SIZE, word_to_ix, 'merge')
    train_data_set = DataLoader(MyDataSet(train_data, word_to_ix, tag_to_ix), batch_size=BATCH_SIZE, shuffle=True)
    # model = torch.load('model/model_merge.pkl')
    model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROP_RATE, len(word_to_ix), tag_to_ix, word_weight)  # 模型
    if use_gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # 优化器

    for epoch in tqdm(range(EPOCH)):
        for _, x in enumerate(train_data_set):
            model.zero_grad()
            sentence_in, target_in = x
            if use_gpu:
                sentence_in = sentence_in.cuda()
                target_in = target_in.cuda()
            # 根据整个数据集进行训练
            loss = model.LossFuction(sentence_in, target_in)
            loss.backward()
            # 梯度下降
            optimizer.step()
    torch.save(model, 'model/model_merge.pkl')
