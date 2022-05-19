# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
from data_preprocess import load_json

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRNN'
        self.label2idx = load_json("./data/label2idx.json")
        self.vocab_path = './data/vocab.txt'  # 词表
        self.num_classes = len(self.label2idx)                         # 类别数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 250                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed =  300                                               # 词向量维度
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''RNN text classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        self.embedding = nn.Embedding(config.n_vocab, config.embed)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.num_layers = 2
        self.hidden_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

    def init_hidden(self,size):
        a = torch.randn(2 * self.num_layers, size, self.hidden_size).to(self.device)
        b = torch.randn(2 * self.num_layers, size, self.hidden_size).to(self.device)
        return (a, b)

    def forward(self, x):
        hidden_ = self.init_hidden(x.shape[0])
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out,hidden_)
        out = torch.sigmoid(self.fc(out[:, -1, :]))  # 句子最后时刻的 hidden state
        return out

    '''变长RNN'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
