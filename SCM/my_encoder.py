#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math

torch.backends.cudnn.enabled = False
class TextCNN(nn.Module):
    def __init__(self, param):
        super(TextCNN, self).__init__()
        # label_num = cfg.label_num # 标签的个数
        self.filter_num = param.filter_num # 卷积核的个数
        self.filter_sizes = [int(fsz) for fsz in param.filter_sizes]
        self.embedding_dim = param.embed_dim
        self.hid_dim = self.embedding_dim

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # if cfg.static: # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        #     self.embedding = self.embedding.from_pretrained(cfg.vectors, freeze=not cfg.fine_tune)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.convs3 = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.convs4 = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.convs5 = nn.ModuleList(
            [nn.Conv2d(1, self.filter_num, (fsz, self.hid_dim )) for fsz in self.filter_sizes])
        self.encoder_dropout = nn.Dropout(param.textcnn_dropout)
        # self.linear = nn.Linear(len(filter_sizes)*filter_num, label_num)

    def forward(self, *x):
        inputs = x[0]
        inputs_n = []
        # x的维度为(batch_size, max_len, embedding_dim)
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        for i,inp in enumerate(inputs):
            inp = inp.view(1, 1, inp.size(1), self.hid_dim)
            # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
            if i==0:
                inp = [F.relu(conv(inp)) for conv in self.convs]
            elif i==1:
                inp = [F.relu(conv1(inp)) for conv1 in self.convs1]
            elif i==2:
                inp = [F.relu(conv2(inp)) for conv2 in self.convs2]
            elif i==3:
                inp = [F.relu(conv3(inp)) for conv3 in self.convs3]
            elif i==4:
                inp = [F.relu(conv4(inp)) for conv4 in self.convs4]
            elif i == 5:
                inp = [F.relu(conv5(inp)) for conv5 in self.convs5]
            # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
            inp = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in inp]
            # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
            inp = [x_item.view(x_item.size(0), -1) for x_item in inp]
            # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
            inp = torch.cat(inp, 1)
            # dropout层
            inp = self.encoder_dropout(inp)
            inputs_n.append(inp.unsqueeze(1))
        output = torch.cat(inputs_n,1)
        # 全连接层
        # logits = self.linear(x)
        return output

class Encoder(nn.Module):
    def __init__(self, param,device):
        super(Encoder, self).__init__()
        self.device = device
        self.bsz = param.batch_size

        self.lstm = nn.LSTM(input_size=param.embed_dim, hidden_size=param.embed_dim, num_layers=2,
                               bias=True, batch_first=True, dropout=0.1, bidirectional=True)
        self.q = nn.Linear(param.embed_dim, param.embed_dim)
        self.k = nn.Linear(param.embed_dim, param.embed_dim)
        self.v = nn.Linear(param.embed_dim, param.embed_dim)

        self.textcnn = TextCNN(param)
        self.filter_num = param.filter_num
        self.part_num = param.part_num
        self.encoder_dim = param.filter_num * len(param.filter_sizes)
        self.cross_margin = param.cross_margin
        self.dropout = nn.Dropout(0.3)

        # self.layernorm = nn.LayerNorm(self.dim, eps=1e-05, elementwise_affine=True)
    def attention(self,query, key, value, mask=None, dropout=None):
        "Scaled Dot Product Attention"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, input, lenths):
        """input: [bsz,seq_len,embedding_dim]"""
        out = torch.zeros(input.size(0),self.part_num, self.encoder_dim).to('cpu')
        for i,inp in enumerate(input):
            q = self.q(inp)
            k = self.k(inp)
            v = self.v(inp)
            inp, inp_att = self.attention(q, k, v)
            # inp = inp[:lenths[i],:].unsqueeze(0)
            # inp,(h,c) = self.lstm(inp)
            inp = inp.unsqueeze(0)
            margin = self.cross_margin
            lenth = math.ceil(lenths[i] // self.part_num)
            if self.part_num ==5:
              part1_inp = inp[:, :lenth, :]
              part2_inp = inp[:, (lenth-margin):(lenth*2+margin), :]
              part3_inp = inp[:, (lenth*2-margin):lenth*3+margin, :]
              part4_inp = inp[:, (lenth*3-margin):(lenth*4+margin), :]
              part5_inp = inp[:, (lenth*4-margin):, :]
              inp = (part1_inp, part2_inp, part3_inp, part4_inp, part5_inp)
            """分3段"""
            if self.part_num == 3:
                part1_inp = inp[:, :lenth, :]
                part2_inp = inp[:, (lenth - margin):(lenth * 2 + margin), :]
                part3_inp = inp[:, (lenth * 2 - margin):, :]
                inp = (part1_inp, part2_inp, part3_inp)
            """分4段"""
            if self.part_num == 4:
                part1_inp = inp[:, :lenth, :]
                part2_inp = inp[:, (lenth-margin):(lenth*2+margin), :]
                part3_inp = inp[:, (lenth*2-margin):lenth*3+margin, :]
                part4_inp = inp[:, (lenth*3-margin):, :]
                inp = (part1_inp, part2_inp, part3_inp, part4_inp)
            if self.part_num ==6:
                part1_inp = inp[:, :lenth, :]
                part2_inp = inp[:, (lenth-margin):(lenth*2+margin), :]
                part3_inp = inp[:, (lenth*2-margin):lenth*3+margin, :]
                part4_inp = inp[:, (lenth*3-margin):(lenth*4+margin), :]
                part5_inp = inp[:, (lenth*4-margin):(lenth*5+margin), :]
                part6_inp = inp[:, (lenth*5-margin):, :]
                inp = (part1_inp, part2_inp, part3_inp, part4_inp, part5_inp, part6_inp)
            cnn_out = self.textcnn(inp)
            out[i] = cnn_out

        return out  # [bsz,part_num,encode_dim]
