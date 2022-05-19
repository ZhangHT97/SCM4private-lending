import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import os
import torch.nn.functional as F

class BiaffineScorer(nn.Module):
    def __init__(self,param,device):
        super().__init__()
        # 为什么+1？？
        # 双仿变换的矩阵形式：
        # S=(H_head⊕1)·W·H_dep
        # 即：(n*m) = (n*(k+1)) * ((k+1)*k) * (k*m)
        self.encoder_dim = param.filter_num * len(param.filter_sizes)
        self.bias = Parameter(torch.Tensor(param.part_num, 1).to(device))
        self.W = Parameter(torch.Tensor(self.encoder_dim+1, self.encoder_dim).to(device))
        nn.init.xavier_normal_(self.bias)
        nn.init.xavier_normal_(self.W)

    def forward(self, input1, input2):
        # input1 size：[bsz,param.part_num, embedding_dim]
        ha_bias = torch.cat((input1, self.bias), 2)
        habias_w = torch.matmul(ha_bias, self.W)
        score = torch.matmul(habias_w, input2.transpose(1,2))
        return score

class PairwiseBilinear(nn.Module):
    '''
    使用版本
    A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)'''

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size
        # W size: [(head_fea_size+1),(dep_fea_size+1),output_size]
        # 无标签弧分类时 output_size=1
        # 标签分类时 output_size=len(labels)
        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0
        nn.init.normal_(self.weight)
        nn.init.normal_(self.bias)

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        # [(batch_size*seq_len),(head_feat_size+1)] * [(head_feat_size+1),((dep_feat_size+1))*output_size]
        # -> [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        intermediate = torch.mm(input1.view(-1, input1_size[-1]),
                                self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        # input2 size: [batch_size, (dep_feat_size+1), seq_len]
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        # intermediate size:
        # [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        #       ->[batch_size, (seq_len*output_size), (dep_feat_size+1)]

        # [batch_size, (seq_len*output_size), (dep_feat_size+1)] * [batch_size, (dep_feat_size+1), seq_len]
        # -> [batch_size, (seq_len*output_size), seq_len]
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        # output size: [batch_size, seq_len, seq_len, output_size]
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)

        return output


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        # 为什么+1？？
        # 双仿变换的矩阵形式：
        # S=(H_head⊕1)·W·H_dep
        # 即：(d*d) = (d*(k+1)) * ((k+1)*k) * (k*d)
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)

        nn.init.normal_(self.W_bilin.weight.data)
        nn.init.normal_(self.W_bilin.bias.data)

    def forward(self, input1, input2):
        # input1 size：[batch_size, seq_len, feature_size]
        # input1.new_ones(*input1.size()[:-1], 1)'s size: [batch_size, seq_len, 1]
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        # 拼接后的size:[batch_size, seq_len, (feature_size+1)]
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class PairwiseBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        """
        使用版本
        :param input1_size:
        :param input2_size:
        :param output_size:双仿的分类空间
        """
        super().__init__()
        # 为什么+1:
        # 双仿变换的矩阵形式：
        # [(batch_size*seq_len),(head_feat_size+1)] * [(head_feat_size+1),((dep_feat_size+1))*output_size]
        #       mm-> [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        # [(batch_size*seq_len),((dep_feat_size+1))*output_size]
        #       view-> [batch_size, (seq_len*output_size), (dep_feat_size+1)]
        # [batch_size, (seq_len*output_size), (dep_feat_size+1)] * [batch_size, (dep_feat_size+1), seq_len]
        #       bmm-> [batch_size, (seq_len*output_size), seq_len]
        # [batch_size, (seq_len*output_size), seq_len]
        #       view-> [batch_size, seq_len, seq_len, output_size]
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)

        nn.init.normal_(self.W_bilin.weight.data)
        nn.init.normal_(self.W_bilin.bias.data)

    def forward(self, input1, input2):
        # input1 size：[batch_size, seq_len, feature_size]
        # input1.new_ones(*input1.size()[:-1], 1)'s size: [batch_size, seq_len, 1]
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        # 拼接后的size:[batch_size, seq_len, (feature_size+1)]
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DirectBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size, pairwise=True):
        super().__init__()
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(input1_size, input2_size, output_size)
        else:
            self.scorer = BiaffineScorer(input1_size, input2_size, output_size)

    def forward(self, input1, input2):
        return self.scorer(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0,
                 pairwise=True):
        """
        使用版本
        :param input1_size:
        :param input2_size:
        :param hidden_size:
        :param output_size: 双仿的分类空间
        :param hidden_func:
        :param dropout:
        :param pairwise:
        """
        super().__init__()
        # 先对输入做两个线性变换得到两个H_dep、H_head
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        # 默认经过relu激活函数：
        self.hidden_func = hidden_func
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        # 进入双仿前dropout:
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))
class TripleMatch(nn.Module):
    def __init__(self, param, vocab_size, encoder, device):
        super(TripleMatch, self).__init__()
        self.max_len = param.max_length
        self.embedding = nn.Embedding(vocab_size, param.embed_dim, padding_idx=0)

        # self.embedding = nn.Embedding(vocab_size, param.embed_dim, padding_idx=0)
        # weights = np.load("data/word2vec_300_dim.embeddings")
        # self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.embed_drop = nn.Dropout(0.2)
        self.embed_layernormal = nn.LayerNorm(param.embed_dim, eps=1e-05, elementwise_affine=True)
        self.device = device
        self.bsz = param.batch_size
        self.encoder = encoder
        self.part_num = param.part_num
        self.encoder_dim = param.filter_num * len(param.filter_sizes)

        self.scorer = DeepBiaffineScorer(self.encoder_dim, self.encoder_dim, self.encoder_dim, 1)
        self.encoder_dim = param.filter_num * len(param.filter_sizes)
        self.W = Parameter(torch.Tensor(self.encoder_dim, self.encoder_dim).to(self.device))
        self.bias = Parameter(torch.Tensor(self.encoder_dim, self.part_num).to(self.device))

        self.layernormal = nn.LayerNorm(self.encoder_dim, eps=1e-05, elementwise_affine=True)
        self.reset_parameters()
        self.loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.margin_loss = nn.MarginRankingLoss(margin=5)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.embedding.weight.data)
        # nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.W, 10)
        nn.init.constant_(self.bias, 10)
        # nn.init.xavier_normal_(self.bias)

    def forward(self,batch,mode="loss"):
        input1, input2, input3, label, lenth = batch
        embed1 = self.embedding(input1)    # [bsz,max_len,embedding_dim]
        embed1 = self.embed_layernormal(embed1)
        embed1 = self.embed_drop(embed1)
        embed2 = self.embedding(input2)
        embed2 = self.embed_layernormal(embed2)
        embed2 = self.embed_drop(embed2)
        embed3 = self.embedding(input3)
        embed3 = self.embed_layernormal(embed3)
        embed3 = self.embed_drop(embed3)

        encoder_out1 = self.encoder(embed1,lenth[:,0])
        # encoder_out1 = self.layernormal(encoder_out1)     #[bsz,max_sent_num,encoder_dim]
        encoder_out2 = self.encoder(embed2,lenth[:,1])
        # encoder_out2 = self.layernormal(encoder_out2)
        encoder_out3 = self.encoder(embed3,lenth[:,2])
        # encoder_out3 = self.layernormal(encoder_out3)

        # position_ids = torch.arange(self.max_len, dtype=torch.long, device=cfg.device)\
        #     .unsqueeze(0).expand(self.bsz,self.max_len)
        # pos_embed = self.pos_embedding(position_ids)
        # pos1_embed = pos_embed[:,:encoder_out1.size(1),:]
        # pos2_embed = pos_embed[:,:encoder_out2.size(1),:]
        # pos3_embed = pos_embed[:,:encoder_out3.size(1),:]
        #
        # inte_inp1 = encoder_out1+pos1_embed
        # inte_inp2 = encoder_out2+pos2_embed
        # inte_inp3 = encoder_out3+pos3_embed

        # score_1 = torch.zeros(self.bsz, 1).to(cfg.device)
        # score_2 = torch.zeros(self.bsz, 1).to(cfg.device)

        # score_b = self.scorer(encoder_out1, encoder_out2).squeeze(-1)
        # score_c = self.scorer(encoder_out1, encoder_out3).squeeze(-1)

        aw = torch.matmul(encoder_out1, self.W)
        awb1 = torch.matmul(aw, encoder_out2.transpose(1, 2))
        bias = torch.matmul(encoder_out1,self.bias)
        awb2 = torch.matmul(aw, encoder_out3.transpose(1, 2))
        score_b = F.relu(awb1 + bias)    # [bsz,sent_a,sent_b]
        score_c = F.relu(awb2 + bias)    # [bsz,sent_a,sent_c]
        score_ab = score_b.view(score_b.size(0), -1).sum(-1)
        score_ac = score_c.view(score_c.size(0), -1).sum(-1)

        # aw = torch.matmul(encoder_out1, self.W)
        # awb1 = F.relu(torch.matmul(aw, encoder_out2.transpose(1, 2)))
        # similarity1 = torch.matmul(encoder_out1, encoder_out2.transpose(1, 2))
        # awb2 = F.relu(torch.matmul(aw, encoder_out3.transpose(1, 2)))
        # similarity2 = torch.matmul(encoder_out2, encoder_out3.transpose(1, 2))

        # score_b = awb1 + similarity1  # [bsz,sent_a,sent_b]
        # score_c = awb2 + similarity2  # [bsz,sent_a,sent_b]

        # score_ab = score_b.sum(-1)  # [bsz,sent_a]
        # score_ac = score_c.sum(-1)  # [bsz,sent_a]
        # score_ab = score_ab.sum(-1)
        # score_ac = score_ac.sum(-1)
        # p1 = self.sigmoid(score_ab-score_ac)
        # p2 = 1.0-p1
        # prob = torch.cat((p1.unsqueeze(1), p2.unsqueeze(1)), -1)
        prob1 = score_ab/(score_ab+score_ac)
        prob2 = score_ac/(score_ab+score_ac)
        prob = torch.cat((prob1.unsqueeze(1), prob2.unsqueeze(1)), -1)
        loss = self.loss(prob.view(-1,2), label.view(-1))
        # label1 = (-(label-1)/2).float()
        # margin_loss = self.margin_loss(score_ab,score_ac,label1)
        if mode=="loss":
            return loss
            # return margin_loss
        if mode=="evaluate":
            return (prob,loss)
            # return (prob, margin_loss)
        if mode=="use_model":
            return score_ab,score_ac









