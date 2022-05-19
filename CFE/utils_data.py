# coding: UTF-8
import os
import torch
import pickle as pkl
import json

MAX_VOCAB_SIZE = 10000  # 词表长度限制
PAD = '<PAD>'
UNK = '<UNK>'

PAD_IDX = 0
UNK_IDX = 1


def build_vocab(min_freq,dict_path):
    vocab_dic = {}
    tmp_dic = {}
    f1 = open("./data/train.json", 'r', encoding='utf-8')
    f2 = open("./data/dev.json", 'r', encoding='utf-8')
    f3 = open("./data/test.json", 'r', encoding='utf-8')
    t1 = f1.readlines()
    t2 = f2.readlines()
    t3 = f3.readlines()
    t = t1+t2+t3
    for d_t in t:
        d_t = json.loads(d_t)
        for word in d_t["text"].split( ):
            num = tmp_dic.get(word, 0) + 1
            tmp_dic[word] = num
    tmp_dict = sorted(tmp_dic.items(), key=lambda d: d[1], reverse=True)
    vocab_dic[PAD] = PAD_IDX
    vocab_dic[UNK] = UNK_IDX
    idx = 2
    for item in tmp_dict:
        if item[1] >= min_freq:
            vocab_dic[item[0]] = idx
            idx += 1
    if not os.path.exists(dict_path):
        with open("./data/dict", 'w', encoding='utf-8') as f:
            for i in vocab_dic:
                f.write(str(i) + '\n')
    return vocab_dic

def build_dataset(config):
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(min_freq=1,dict_path="./data/vocab.txt")
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")
    f_label = open("./data/label2idx.json", 'r', encoding='utf-8')
    labelid_list = json.loads(f_label.read())
    def load_dataset(path, pad_size=config.pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            ft = f.readlines()
            for line in ft:
                line = json.loads(line)
                fact = line["text"].split( )
                words_line = []
                seq_len = len(fact)
                if pad_size:
                    if len(fact) < pad_size:
                        fact.extend([PAD] * (pad_size - len(fact)))
                    else:
                        fact = fact[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in fact:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                tmp_label = [0] * len(labelid_list)
                label_list = line["label"]
                for l in label_list:
                    idc = labelid_list.get(l, 1)
                    tmp_label[idc] = 1
                contents.append((words_line, tmp_label, seq_len))
        return contents
    train = load_dataset("./data/train.json")
    dev = load_dataset("./data/dev.json")
    test = load_dataset("./data/test.json")
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter



# if __name__ == "__main__":
#     # build_vocab(5,"./data/vocab.txt")
#     build_dataset("./data/vocab.txt")
#
#     '''提取预训练词向量'''
#     # 下面的目录、文件名按需更改。
#     train_dir = "./THUCNews/data/train.txt"
#     vocab_dir = "./THUCNews/data/vocab.pkl"
#     pretrain_dir = "./THUCNews/data/sgns.sogou.char"
#     emb_dim = 300
#     filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
#     if os.path.exists(vocab_dir):
#         word_to_id = pkl.load(open(vocab_dir, 'rb'))
#     else:
#         # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
#         tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
#         word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
#         pkl.dump(word_to_id, open(vocab_dir, 'wb'))
#
#     embeddings = np.random.rand(len(word_to_id), emb_dim)
#     f = open(pretrain_dir, "r", encoding='UTF-8')
#     for i, line in enumerate(f.readlines()):
#         # if i == 0:  # 若第一行是标题，则跳过
#         #     continue
#         lin = line.strip().split(" ")
#         if lin[0] in word_to_id:
#             idx = word_to_id[lin[0]]
#             emb = [float(x) for x in lin[1:301]]
#             embeddings[idx] = np.asarray(emb, dtype='float32')
#     f.close()
#     np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
