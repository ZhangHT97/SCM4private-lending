# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from BERT_Label import Bert_label
from label_helper import BrtLabDataSet
from sklearn.metrics import accuracy_score,f1_score
from sklearn import metrics
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

train_path = "./bertlabel_data/train.json"
dev_path = "./bertlabel_data/dev.json"
test_path = "./bertlabel_data/test.json"
label2idx_path = "./bertlabel_data/label2idx.json"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)
device = "cuda" if torch.cuda.is_available() else "cpu"

# congfig for bert
learning_rate = 1e-4
batch_size = 3
max_len = 512
hidden_size = 768
epochs_num = 20

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_acc_score(logits,labels):
    max_id0 = logits[0].argmax(dim=1)
    max_id1 = logits[1].argmax(dim=1)
    max_id2 = logits[2].argmax(dim=1)
    max_id3 = logits[3].argmax(dim=1)
    max_id4 = logits[4].argmax(dim=1)
    max_id5 = logits[5].argmax(dim=1)
    max_id6 = logits[6].argmax(dim=1)
    max_id7 = logits[7].argmax(dim=1)
    max_id8 = logits[8].argmax(dim=1)
    max_id9 = logits[9].argmax(dim=1)
    pred = []
    pred.append(torch.zeros(logits[0].shape).to(device))
    pred.append(torch.zeros(logits[1].shape).to(device))
    pred.append(torch.zeros(logits[2].shape).to(device))
    pred.append(torch.zeros(logits[3].shape).to(device))
    pred.append(torch.zeros(logits[4].shape).to(device))
    pred.append(torch.zeros(logits[5].shape).to(device))
    pred.append(torch.zeros(logits[6].shape).to(device))
    pred.append(torch.zeros(logits[7].shape).to(device))
    pred.append(torch.zeros(logits[8].shape).to(device))
    pred.append(torch.zeros(logits[9].shape).to(device))
    for i in range(logits[0].size(0)):
        pred[0][i].index_fill_(-1,max_id0[i],1)
        pred[1][i].index_fill_(-1,max_id1[i],1)
        pred[2][i].index_fill_(-1,max_id2[i],1)
        pred[3][i].index_fill_(-1,max_id3[i],1)
        pred[4][i].index_fill_(-1,max_id4[i],1)
        pred[5][i].index_fill_(-1,max_id5[i],1)
        pred[6][i].index_fill_(-1,max_id6[i],1)
        pred[7][i].index_fill_(-1,max_id7[i],1)
        pred[8][i].index_fill_(-1,max_id8[i],1)
        pred[9][i].index_fill_(-1,max_id9[i],1)
    pred_label = torch.cat(pred,dim=-1)
    y_pred = pred_label.cpu().detach().numpy()
    y_true = labels.cpu().detach().numpy()
    micro_f1 = f1_score(y_true,y_pred,average="micro")
    macro_f1 = f1_score(y_true,y_pred,average="macro")
    average = (micro_f1 + macro_f1) / 2
    return micro_f1,macro_f1,average

def loss_sum(logits,labels,loss_fc):
    la_len = load_json("./bertlabel_data/label_len")
    len1_id = la_len[0]
    len2_id = len1_id + la_len[1]
    len3_id = len2_id + la_len[2]
    len4_id = len3_id + la_len[3]
    len5_id = len4_id + la_len[4]
    len6_id = len5_id + la_len[5]
    len7_id = len6_id + la_len[6]
    len8_id = len7_id + la_len[7]
    len9_id = len8_id + la_len[8]
    len10_id = len9_id + la_len[9]
    label0 = labels[:,:len1_id]
    label1 = labels[:,len1_id:len2_id]
    label2 = labels[:,len2_id:len3_id]
    label3 = labels[:,len3_id:len4_id]
    label4 = labels[:,len4_id:len5_id]
    label5 = labels[:,len5_id:len6_id]
    label6 = labels[:,len6_id:len7_id]
    label7 = labels[:,len7_id:len8_id]
    label8 = labels[:,len8_id:len9_id]
    label9 = labels[:,len9_id:len10_id]
    loss0 = loss_fc(logits[0],label0)
    loss1 = loss_fc(logits[1], label1)
    loss2 = loss_fc(logits[2], label2)
    loss3 = loss_fc(logits[3], label3)
    loss4 = loss_fc(logits[4], label4)
    loss5 = loss_fc(logits[5], label5)
    loss6 = loss_fc(logits[6], label6)
    loss7 = loss_fc(logits[7], label7)
    loss8 = loss_fc(logits[8], label8)
    loss9 = loss_fc(logits[9], label9)
    loss = (loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)/10
    return loss

def train(args):
    model = Bert_label(hidden_size=hidden_size, class_num=class_num,args=args)
    lr = learning_rate
    epochs = epochs_num
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.001)     # 设置学习率下降策略
    dev_best_f1 = 0
    start_epoch = 0
    for epoch in range(start_epoch+1, epochs):
      model.train()
      for i, batch in enumerate(train_dataloader):
          input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
          optimizer.zero_grad()
          logits = model(input_ids, token_type_ids, attention_mask)
          loss = loss_sum(logits, labels,criterion)
          loss.backward()
          optimizer.step()
          if i % 100 == 0:
            micro_f1,macro_f1,average = get_acc_score(logits, labels)
            print("***********************************************")
            print("Micro_f1:%.3f Macro_f1:%.3f Average:%.3f Loss:%.3f"%(micro_f1,macro_f1,average,loss))
            print("***********************************************")
            # print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))
      scheduler.step()
      # 验证集
      dev_loss, dev_mic_f1,dev_mac_f1,dev_ave_f1 = dev(model, dev_dataloader, criterion)
      print("Dev epoch:{} mic_f1:{} mac_f1:{} average_f1:{} loss:{}".format(epoch,dev_mic_f1,dev_mac_f1,dev_ave_f1,dev_loss))
      if dev_ave_f1 > dev_best_f1:
          dev_best_f1 = dev_ave_f1
          torch.save(model.state_dict(), args.model_path)

    # 测试
    tes_micro_f1,tes_macro_f1,test_average = tes(args)
    print("Test mic_f1:{} mac_f1:{} average_f1:{}".format(tes_micro_f1,tes_macro_f1,test_average))


def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    micro_f1_list = []
    macro_f1_list = []
    ave_f1_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = loss_sum(logits, labels, criterion)
            all_loss.append(loss.item())
            micro_f1, macro_f1, average = get_acc_score(logits,labels)
            micro_f1_list.append(micro_f1)
            macro_f1_list.append(macro_f1)
            ave_f1_list.append(average)
    # print('***********分类报告*************\n', metrics.classification_report(true_labels, pred_labels))
    return np.mean(all_loss),np.mean(micro_f1_list),np.mean(macro_f1_list),np.mean(ave_f1_list)

def tes(args):
    model = Bert_label(hidden_size=hidden_size, class_num=class_num,args=args)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    micro_f1_list = []
    macro_f1_list = []
    ave_f1_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits = model(input_ids, token_type_ids, attention_mask)
            micro_f1, macro_f1, average = get_acc_score(logits, labels)
            micro_f1_list.append(micro_f1)
            macro_f1_list.append(macro_f1)
            ave_f1_list.append(average)
            # print('***********分类报告*************\n', metrics.classification_report(true_labels, pred_labels))
    return np.mean(micro_f1_list), np.mean(macro_f1_list), np.mean(ave_f1_list)
    # acc_score = get_acc_score(true_labels, pred_labels)
    # return acc_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--model_name', type=str, default="BertLabel")
    parser.add_argument('--encode_dim', type=int, default=768,
                        help="bert encode_dim")
    parser.add_argument('--pad_size', type=int, default=512,
                        help="max lenth")
    parser.add_argument('--num_filters', type=int, default=256,
                        help="卷积核数量(channels数)")
    parser.add_argument('--a', type=float, default=1,
                        help="merge a")
    parser.add_argument('--b', type=float, default=1,
                        help="merge b")
    args = parser.parse_args()
    args.model_path = "./model/{}.pkl".format(args.model_name)
    set_seed(args)
    label2idx = load_json(label2idx_path)
    class_num = len(label2idx)
    train_dataset = BrtLabDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
    dev_dataset = BrtLabDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = BrtLabDataSet(test_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train(args)