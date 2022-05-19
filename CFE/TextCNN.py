# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocess import load_json

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.label2idx = load_json("./data/label2idx.json")
        self.vocab_path ='./data/vocab.txt'                                # 词表
        # self.save_path ='./saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.num_classes = len(self.label2idx)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.embed = 300                                                # embedding_dim
        self.pad_size = 300                                              # 每句话处理成的长度(短填长切)
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.label_lenth = load_json("./data/label_len")


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.label2idx = config.label2idx
        self.label_lenth = config.label_lenth

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
