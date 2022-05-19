# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from data_preprocess import load_json
from transformers import BertModel,BertTokenizer

class bert_lstm(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(bert_lstm, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("./bert_model")

        self.label2idx = load_json("./data/label2idx.json")
        self.num_classes = len(self.label2idx)  # 类别数
        self.encode_dim = 768  # embedding_dim
        self.num_layers = 2  # lstm层数
        self.dropout = 0.5                                              # 随机失活
        self.hidden_size = 384
        self.lstm = nn.LSTM(self.encode_dim, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)


    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_out = self.bert(input_ids, token_type_ids, attention_mask)
        out = self.drop(encoder_out[0])
        out, _ = self.lstm(out)
        out = torch.sigmoid(self.fc(out[:, -1, :]))  # 句子最后时刻的 hidden state
        return out