from torch.utils.data import Dataset, DataLoader
import torch


# 根据to_ix将列表转化成Tensor
def Word2Tensor(word_set, to_ix):
    return torch.tensor([to_ix[i] if i in to_ix.keys() else 1 for i in word_set], dtype=torch.long)


# batch版 根据to_ix将列表转化成Tensor
def Word2TensorBatch(train_set, word_to_ix, tag_to_ix):
    sentence_batch = torch.tensor([[word_to_ix[word] for word in line[0]] for line in train_set], dtype=torch.long)
    tags_batch = torch.tensor([[tag_to_ix[tag] for tag in line[1]] for line in train_set], dtype=torch.long)
    return sentence_batch, tags_batch


class MyDataSet(Dataset):
    def __init__(self, data_from, word_to_ix, tag_to_ix):
        self.sentence, self.tag = Word2TensorBatch(data_from, word_to_ix, tag_to_ix)

    def __getitem__(self, item):
        return self.sentence[item], self.tag[item]

    def __len__(self):
        return len(self.sentence)
