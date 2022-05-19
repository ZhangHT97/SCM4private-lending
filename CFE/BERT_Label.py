# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from data_preprocess import load_json
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer

class Bert_label(nn.Module):
    def __init__(self, hidden_size, class_num,args,dropout=0.5):
        super(Bert_label, self).__init__()
        self.fc = nn.Linear(hidden_size, class_num)
        self.drop = nn.Dropout(dropout)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert = BertModel.from_pretrained("./bert_model")
        self.tokenizer = BertTokenizer.from_pretrained("./bert_model/")  # 本地vocab
        self.label2idx = load_json("./bertlabel_data/label2idx.json")
        self.label_lenth = load_json("./bertlabel_data/label_len")
        self.num_classes = len(self.label2idx)  # 类别数
        self.encode_dim = args.encode_dim  # embedding_dim
        self.pad_size = args.pad_size  # 每句话处理成的长度(短填长切)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = args.num_filters  # 卷积核数量(channels数)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.encode_dim)) for k in self.filter_sizes])
        self.a = args.a
        self.b = args.b
        self.merge = nn.Linear(args.encode_dim,args.encode_dim)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(10,args.encode_dim)
        self.norml = nn.BatchNorm1d(1,args.encode_dim)
        # self.fc = nn.Linear(7680, self.num_classes)
#         self.classifier = nn.Sequential(nn.Linear(self.encode_dim*10,self.encode_dim),
#                                         nn.Linear(self.encode_dim,self.num_classes))
#         self.classifier0 = nn.Linear(self.encode_dim*2,self.label_lenth[0])
#         self.classifier1 = nn.Linear(self.encode_dim*2, self.label_lenth[1])
#         self.classifier2 = nn.Linear(self.encode_dim*2, self.label_lenth[2])
#         self.classifier3 = nn.Linear(self.encode_dim*2, self.label_lenth[3])
#         self.classifier4 = nn.Linear(self.encode_dim*2, self.label_lenth[4])
#         self.classifier5 = nn.Linear(self.encode_dim*2, self.label_lenth[5])
#         self.classifier6 = nn.Linear(self.encode_dim*2, self.label_lenth[6])
#         self.classifier7 = nn.Linear(self.encode_dim*2, self.label_lenth[7])
#         self.classifier8 = nn.Linear(self.encode_dim*2, self.label_lenth[8])
#         self.classifier9 = nn.Linear(self.encode_dim*2, self.label_lenth[9])
        self.classifier0 = nn.Sequential(nn.Linear(self.encode_dim*2,self.label_lenth[0]),nn.Sigmoid())
        self.classifier1 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[1]),nn.Sigmoid())
        self.classifier2 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[2]),nn.Sigmoid())
        self.classifier3 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[3]),nn.Sigmoid())
        self.classifier4 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[4]),nn.Sigmoid())
        self.classifier5 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[5]),nn.Sigmoid())
        self.classifier6 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[6]),nn.Sigmoid())
        self.classifier7 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[7]),nn.Sigmoid())
        self.classifier8 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[8]),nn.Sigmoid())
        self.classifier9 = nn.Sequential(nn.Linear(self.encode_dim*2, self.label_lenth[9]),nn.Sigmoid())

    def encode_attr(self):
        attr_label = load_json("./bertlabel_data/attrlist")
        attr_list = list(attr_label.keys())
        label_all = list(attr_label.values())
        label_list = []
        for i in range(len(label_all)):
            label_list += label_all[i]
        attr_list_id = [self.tokenizer.encode(text,max_length=8,return_tensors="pt").to(self.device) for text in attr_list]
        label_list_id = [self.tokenizer.encode(text,max_length=15,return_tensors="pt").to(self.device) for text in label_list]
        encode_attr_list = []
        encode_label_list = []
        for i,attr_inp in enumerate(attr_list_id):
            encode_attr_list.append(self.bert(attr_inp)["last_hidden_state"].mean(dim=1))
        encode_attr = torch.cat(encode_attr_list,0)
        for i,label_inp in enumerate(label_list_id):
            encode_label_list.append(self.bert(label_inp)["last_hidden_state"].mean(dim=1))
        encode_label = torch.cat(encode_label_list,0)
        return encode_attr, encode_label   # [10,dim],[num_label,dim]

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def label_att(self,h_label,h_fact):
        """
        input0:[label_num,dim]
        input1:[bsz,10,dim]
        """
        h_fact = h_fact.transpose(1,2) #[bsz,dim,10]
        attr_la = torch.matmul(h_label,h_fact)
        alpha = F.softmax(attr_la,dim=1).transpose(1,2)
        out = torch.matmul(alpha,h_label)  #[bsz,10,dim]
        return out

    def forward(self, input_ids, token_type_ids, attention_mask):
        encode_attr, encode_label = self.encode_attr()
        encoder_fact = self.bert(input_ids, token_type_ids, attention_mask)
        fact_pool = self.drop(encoder_fact[1].unsqueeze(1).expand(-1,10,-1))
        fact = self.drop(encoder_fact[0])
        fact = fact.unsqueeze(1)
        fact = torch.cat([self.conv_and_pool(fact, conv) for conv in self.convs], 1) # [bsz,dim]
        encode_fact = fact.unsqueeze(1)
        fact_attr = self.merge(self.a * encode_fact + self.b * encode_attr) # [bsz,10,dim]
        la_out = self.label_att(encode_label,fact_attr)
#         la_out = self.norm(la_out)
#         print(la_out.shape)
#         print(fact_pool.shape)
        out = torch.cat((la_out,fact_pool),-1)
        y0 = self.classifier0(out[:,0,:])
        y1 = self.classifier1(out[:,1,:])
        y2 = self.classifier2(out[:,2,:])
        y3 = self.classifier3(out[:,3,:])
        y4 = self.classifier4(out[:,4,:])
        y5 = self.classifier5(out[:,5,:])
        y6 = self.classifier6(out[:,6,:])
        y7 = self.classifier7(out[:,7,:])
        y8 = self.classifier8(out[:,8,:])
        y9 = self.classifier9(out[:,9,:])
        out = torch.cat((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9),-1)
#         out = torch.sigmoid(out)
        # out = out.view(-1,7680)
        # out = torch.sigmoid(self.classifier(out))
        return out,(y0,y1,y2,y3,y4,y5,y6,y7,y8,y9)












