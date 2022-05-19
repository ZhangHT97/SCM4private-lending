# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from data_preprocess import load_json
from transformers import BertModel,BertTokenizer
from transformers import RobertaModel,RobertaTokenizer

class Bert_label(nn.Module):
    def __init__(self, hidden_size, class_num,args,dropout=0.1):
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
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.classifier0 = nn.Sequential(nn.Linear(self.encode_dim,self.label_lenth[0]),
                                         nn.Softmax(dim=-1))
        self.classifier1 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[1]),
                                         nn.Softmax(dim=-1))
        self.classifier2 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[2]),
                                         nn.Softmax(dim=-1))
        self.classifier3 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[3]),
                                         nn.Softmax(dim=-1))
        self.classifier4 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[4]),
                                         nn.Softmax(dim=-1))
        self.classifier5 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[5]),
                                         nn.Softmax(dim=-1))
        self.classifier6 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[6]),
                                         nn.Softmax(dim=-1))
        self.classifier7 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[7]),
                                         nn.Softmax(dim=-1))
        self.classifier8 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[8]),
                                         nn.Softmax(dim=-1))
        self.classifier9 = nn.Sequential(nn.Linear(self.encode_dim, self.label_lenth[9]),
                                         nn.Softmax(dim=-1))

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
        la_lenth = self.label_lenth
        la1_id = la_lenth[0]
        la2_id = la1_id+la_lenth[1]
        la3_id = la2_id+la_lenth[2]
        la4_id = la3_id+la_lenth[3]
        la5_id = la4_id+la_lenth[4]
        la6_id = la5_id+la_lenth[5]
        la7_id = la6_id+la_lenth[6]
        la8_id = la7_id+la_lenth[7]
        la9_id = la8_id+la_lenth[8]
        la10_id = la9_id+la_lenth[9]
        h_fact = h_fact.transpose(1,2) #[bsz,dim,10]
        attr1_la = torch.matmul(h_label[:la1_id,],h_fact[:,:,0].unsqueeze(-1))
        attr2_la = torch.matmul(h_label[la1_id:la2_id,: ], h_fact[:,:,1].unsqueeze(-1)) #h_label:[n,dim]
        attr3_la = torch.matmul(h_label[la2_id:la3_id,: ], h_fact[:,:,2].unsqueeze(-1))
        attr4_la = torch.matmul(h_label[la3_id:la4_id,: ], h_fact[:,:,3].unsqueeze(-1))
        attr5_la = torch.matmul(h_label[la4_id:la5_id,: ], h_fact[:,:,4].unsqueeze(-1))
        attr6_la = torch.matmul(h_label[la5_id:la6_id,: ], h_fact[:,:,5].unsqueeze(-1))
        attr7_la = torch.matmul(h_label[la6_id:la7_id,: ], h_fact[:,:,6].unsqueeze(-1))
        attr8_la = torch.matmul(h_label[la7_id:la8_id,: ], h_fact[:,:,7].unsqueeze(-1))
        attr9_la = torch.matmul(h_label[la8_id:la9_id,: ], h_fact[:,:,8].unsqueeze(-1))
        attr10_la = torch.matmul(h_label[la9_id:la10_id,:], h_fact[:,:,9].unsqueeze(-1))
        alpha1 = F.softmax(attr1_la,dim=1).transpose(1,2)  #[bsz,1,n]
        alpha2 = F.softmax(attr2_la,dim=1).transpose(1,2)
        alpha3 = F.softmax(attr3_la,dim=1).transpose(1,2)
        alpha4 = F.softmax(attr4_la,dim=1).transpose(1,2)
        alpha5 = F.softmax(attr5_la,dim=1).transpose(1,2)
        alpha6 = F.softmax(attr6_la,dim=1).transpose(1,2)
        alpha7 = F.softmax(attr7_la,dim=1).transpose(1,2)
        alpha8 = F.softmax(attr8_la,dim=1).transpose(1,2)
        alpha9 = F.softmax(attr9_la,dim=1).transpose(1,2)
        alpha10 = F.softmax(attr10_la,dim=1).transpose(1,2)
        out1 = torch.matmul(alpha1,h_label[:la1_id,]).squeeze(1)  #[bsz,1,dim]
        out2 = torch.matmul(alpha2,h_label[la1_id:la2_id, ]).squeeze(1)
        out3 = torch.matmul(alpha3,h_label[la2_id:la3_id, ]).squeeze(1)
        out4 = torch.matmul(alpha4,h_label[la3_id:la4_id, ]).squeeze(1)
        out5 = torch.matmul(alpha5,h_label[la4_id:la5_id, ]).squeeze(1)
        out6 = torch.matmul(alpha6,h_label[la5_id:la6_id, ]).squeeze(1)
        out7 = torch.matmul(alpha7,h_label[la6_id:la7_id, ]).squeeze(1)
        out8 = torch.matmul(alpha8,h_label[la7_id:la8_id, ]).squeeze(1)
        out9 = torch.matmul(alpha9,h_label[la8_id:la9_id, ]).squeeze(1)
        out10 = torch.matmul(alpha10,h_label[la9_id:la10_id,]).squeeze(1)
        return (out1,out2,out3,out4,out5,out6,out7,out8,out9,out10)

    def forward(self, input_ids, token_type_ids, attention_mask):
        encode_attr, encode_label = self.encode_attr()
        encoder_fact = self.bert(input_ids, token_type_ids, attention_mask)
        fact = self.drop(encoder_fact[0])
        fact = fact.unsqueeze(1)
        fact = torch.cat([self.conv_and_pool(fact, conv) for conv in self.convs], 1) # [bsz,dim]
        encode_fact = torch.unsqueeze(fact,1)
        fact_attr = self.merge(self.a * encode_fact + self.b * encode_attr) # [bsz,10,dim]
        # out = fact_attr
        out = self.label_att(encode_label,fact_attr)
        y0 = self.classifier0(out[0])
        y1 = self.classifier1(out[1])
        y2 = self.classifier2(out[2])
        y3 = self.classifier3(out[3])
        y4 = self.classifier4(out[4])
        y5 = self.classifier5(out[5])
        y6 = self.classifier6(out[6])
        y7 = self.classifier7(out[7])
        y8 = self.classifier8(out[8])
        y9 = self.classifier9(out[9])
        return (y0,y1,y2,y3,y4,y5,y6,y7,y8,y9)












