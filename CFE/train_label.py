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
    tem_logits = (logits.cpu() > 0.5).int().detach().numpy()
    out_label_ids = labels.cpu().detach().numpy()
    y_true = []
    y_pred = []
    for i in range(len(tem_logits)):
        tem_1 = tem_logits[i]
        tem_2 = out_label_ids[i]
        y_pred.append(list(tem_1))
        y_true.append(list(tem_2))
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    average = (micro_f1 + macro_f1) / 2
    return micro_f1, macro_f1, average

def train(args):
    f1 = open('Bert_label_log.txt','w',encoding='utf-8')
    model = Bert_label(hidden_size=hidden_size, class_num=class_num,args=args)
    lr = learning_rate
    epochs = epochs_num
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.001)
    dev_best_f1 = 0
    start_epoch = 0
    for epoch in range(start_epoch+1, epochs):
      model.train()
      for i, batch in enumerate(train_dataloader):
          input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
          optimizer.zero_grad()
          logits,all_logits = model(input_ids, token_type_ids, attention_mask)
          loss = criterion(logits, labels)
          loss.backward()
          optimizer.step()
          micro_f1, macro_f1, average = get_acc_score(logits, labels)
          if i % 100 == 0:
            micro_f1,macro_f1,average = get_acc_score(logits, labels)
            print("***********************************************")
            print("Micro_f1:%.3f Macro_f1:%.3f Average:%.3f Loss:%.3f"%(micro_f1,macro_f1,average,loss))
            print("***********************************************")
            f1.write("Micro_f1:%.3f Macro_f1:%.3f Average:%.3fLoss:%.3f"%(micro_f1,macro_f1,average,loss)+"\n")
            # print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))
      scheduler.step()
      # 验证集
      dev_loss, dev_mic_f1,dev_mac_f1,dev_ave_f1 = dev(model, dev_dataloader, criterion)
      print("Dev epoch:{} mic_f1:{} mac_f1:{} average_f1:{} loss:{}".format(epoch,dev_mic_f1,dev_mac_f1,dev_ave_f1,dev_loss))
      f1.write("Dev epoch:{} mic_f1:{} mac_f1:{} average_f1:{} loss:{}".format(epoch,dev_mic_f1,dev_mac_f1,dev_ave_f1,dev_loss,) + "\n")
      if dev_ave_f1 > dev_best_f1:
          dev_best_f1 = dev_ave_f1
          torch.save(model.state_dict(), args.model_path)

    # 测试
    tes_micro_f1,tes_macro_f1,test_average = tes(args)
    print("Test mic_f1:{} mac_f1:{} average_f1:{}".format(tes_micro_f1,tes_macro_f1,test_average))
    f1.write("Test mic_f1:{} mac_f1:{} average_f1:{}".format(tes_micro_f1,tes_macro_f1,test_average) + "\n")
    f1.close()


def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    micro_f1_list = []
    macro_f1_list = []
    ave_f1_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
            logits, all_logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
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
            logits, all_logits = model(input_ids, token_type_ids, attention_mask)
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