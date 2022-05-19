# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from data_preprocess import load_json
from transformers import BertModel,BertTokenizer

class bert_cnn(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(bert_cnn, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("./bert_model")
        self.label2idx = load_json("./data/label2idx.json")
        self.num_classes = len(self.label2idx)  # 类别数
        self.encode_dim = 768  # embedding_dim
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.encode_dim)) for k in self.filter_sizes])
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.label2idx = load_json("./bert_data/label2idx.json")
        self.label_lenth = load_json("./bert_data/label_len")

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_out = self.bert(input_ids, token_type_ids, attention_mask)
        out = self.drop(encoder_out[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = torch.sigmoid(self.fc(out))
        return out
