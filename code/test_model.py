import torch
from tqdm import tqdm
from main import LoadTestData, Word2Tensor,Tag2Word, tag_to_ix, LoadData, GetF1

torch.manual_seed(1)
use_gpu = torch.cuda.is_available()

data = LoadData(r'msrseg/msr_training.utf8', False, 200)
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
model = torch.load('model/model_merge.pkl')
if use_gpu:
    model = model.cuda()
with torch.no_grad():
    sentences_f1 = 0.0
    for i in tqdm(range(len(test_data))):
        sentence = Word2Tensor(test_data[i][0], word_to_ix)
        if use_gpu:
            sentence = sentence.cuda()
        model_result_tag = model(sentence)[1]
        model_result = Tag2Word(test_data[i][0], model_result_tag, tag_to_ix, True)
        true_result = Tag2Word(test_data[i][0], test_data[i][1], tag_to_ix)
        sentences_f1 += GetF1(model_result, true_result)
        with open('result/model_merge_result.txt', 'a', encoding='utf-8') as fp:
            fp.write(" ".join(model_result) + '\n')
    print(sentences_f1 / len(test_data))