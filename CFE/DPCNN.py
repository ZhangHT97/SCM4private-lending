# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_preprocess import load_json


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'DPCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.label2idx = load_json("./data/label2idx.json")
        self.vocab_path = './data/vocab.txt'  # 词表
        self.dropout = 0                                             # 随机失活
        self.num_classes = len(self.label2idx)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 30                                            # epoch数
        self.pad_size = 300                                              # 每句话处理成的长度(短填长切)
        self.batch_size = 32                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300  # embedding_dim
        self.num_filters = 250                                          # 卷积核数量(channels数)

'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters*2, config.num_classes)

    def forward(self, x):
        x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze().view(x.size()[0],-1)  # [batch_size, num_filters(250)*2]
        x = torch.sigmoid(self.fc(x))
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
