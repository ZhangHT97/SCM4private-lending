# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from BERT_Label import Bert_label
from roberta import Roberta
from bert_cnn import bert_cnn
from bert_lstm import bert_lstm
from data_helper import MultiClsDataSet
from roberta_helper import RobertaDataSet
from sklearn.metrics import accuracy_score,f1_score
from sklearn import metrics
import argparse
import random
from importlib import import_module
import warnings
warnings.filterwarnings("ignore")

train_path = "./data/train.json"
dev_path = "./data/dev.json"
test_path = "./data/test.json"
label2idx_path = "./data/label2idx.json"
# save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)
device = "cuda" if torch.cuda.is_available() else "cpu"

# congfig for bert
learning_rate = 4e-5
batch_size = 3
max_len = 512
hidden_size = 768
epochs_num = 15

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_acc_score(y_pred_tensor,y_true_tensor):
  tem_logits = (y_pred_tensor.cpu() > 0.5).int().detach().numpy()
  out_label_ids = y_true_tensor.cpu().detach().numpy()
  y_true = []
  y_pred = []
  for i in range(len(tem_logits)):
    tem_1=tem_logits[i]
    tem_2=out_label_ids[i]
    y_pred.append(list(tem_1))
    y_true.append(list(tem_2))
  micro_f1 = f1_score(y_true,y_pred,average="micro")
  macro_f1 = f1_score(y_true,y_pred,average="macro")
  average = (micro_f1 + macro_f1) / 2
  return micro_f1,macro_f1,average

def train(args):
    f1 = open(args.model_name+'log.txt','w',encoding='utf-8')
    if args.model_name in ["TextCNN","DPCNN","TextRNN","TextRNN_Att","FastText"]:
        model = x.Model(config).to(config.device)
        lr = config.learning_rate
        epochs = config.num_epochs
    elif args.model_name == "bert-cnn":
        model = bert_cnn(hidden_size=hidden_size, class_num=class_num)
        lr = learning_rate
        epochs = epochs_num
    elif args.model_name == "bert-lstm":
        model = bert_lstm(hidden_size=hidden_size, class_num=class_num)
        lr = learning_rate
        epochs = epochs_num
    elif args.model_name == "roberta":
        model = Roberta(hidden_size=hidden_size, class_num=class_num)
        lr = learning_rate
        epochs = epochs_num
    else:
        model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
        lr = learning_rate
        epochs = epochs_num
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略

    dev_best_f1 = 0.
    start_epoch = 0

    for epoch in range(start_epoch+1, epochs):
      model.train()
      for i, batch in enumerate(train_dataloader):
          if args.model_name in ["TextCNN", "DPCNN", "TextRNN","TextRNN_Att","FastText"]:
              input,labels = batch
              labels = labels.float()
              optimizer.zero_grad()
              logits = model(input)
          elif args.model_name == "roberta":
              input_ids, attention_mask, labels = [d.to(device) for d in batch]
              optimizer.zero_grad()
              logits = model(input_ids, attention_mask)
          else:
              input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
              optimizer.zero_grad()
              logits = model(input_ids, token_type_ids, attention_mask)
          loss = criterion(logits, labels)
          loss.backward()
          optimizer.step()
          if i % 100 == 0:
            micro_f1,macro_f1,average = get_acc_score(logits, labels)
            print("***********************************************")
            print("Micro_f1:%.3f Macro_f1:%.3f Average:%.3f Loss:%.3f"%(micro_f1,macro_f1,average,loss))
            print("***********************************************")
            f1.write("Micro_f1:%.3f Macro_f1:%.3f Average:%.3f Loss:%.3f"%(micro_f1,macro_f1,average,loss)+"\n")
            # print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))
      scheduler.step()

      # 验证集合
      dev_loss, dev_mic_f1,dev_mac_f1,dev_ave_f1 = dev(model, dev_dataloader, criterion)
      print("Dev epoch:{} mic_f1:{} mac_f1:{} average_f1:{} loss:{}".format(epoch,dev_mic_f1,dev_mac_f1,dev_ave_f1,dev_loss,))
      f1.write("Dev epoch:{} mic_f1:{} mac_f1:{} average_f1:{} loss:{}".format(epoch,dev_mic_f1,dev_mac_f1,dev_ave_f1,dev_loss,) + "\n")
      if dev_ave_f1 > dev_best_f1:
          dev_best_f1 = dev_ave_f1
          torch.save(model.state_dict(), args.model_path)

    # 测试
    tes_micro_f1,tes_macro_f1,test_average = test(args)
    print("Test mic_f1:{} mac_f1:{} average_f1:{}".format(tes_micro_f1,tes_macro_f1,test_average))
    f1.write("Test mic_f1:{} mac_f1:{} average_f1:{}".format(tes_micro_f1,tes_macro_f1,test_average) + "\n")
    f1.close()

def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if args.model_name in ["TextCNN", "DPCNN", "TextRNN","TextRNN_Att","FastText"]:
                input, labels = batch
                labels = labels.float()
                logits = model(input)
            elif args.model_name == "roberta":
                input_ids, attention_mask, labels = [d.to(device) for d in batch]
                logits = model(input_ids, attention_mask)
            else:
                input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
                logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    # acc_score = get_acc_score(true_labels, pred_labels)
    micro_f1,macro_f1,average = get_acc_score(pred_labels,true_labels)
    # print('***********分类报告*************\n', metrics.classification_report(true_labels, pred_labels))
    return np.mean(all_loss),micro_f1,macro_f1,average

def test(args):
    if args.model_name in ["TextCNN","DPCNN","TextRNN","TextRNN_Att","FastText"]:
        model = x.Model(config).to(config.device)
    elif args.model_name == "bert-cnn":
        model = bert_cnn(hidden_size=hidden_size, class_num=class_num)
    elif args.model_name == "bert-lstm":
        model = bert_lstm(hidden_size=hidden_size, class_num=class_num)
    elif args.model_name == "roberta":
        model = Roberta(hidden_size=hidden_size, class_num=class_num)
    else:
        model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if args.model_name in ["TextCNN","DPCNN","TextRNN","TextRNN_Att","FastText"]:
                input, labels = batch
                labels = labels.float()
                logits = model(input)
            elif args.model_name == "roberta":
                input_ids, attention_mask, labels = [d.to(device) for d in batch]
                logits = model(input_ids, attention_mask)
            else:
                input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in batch]
                logits = model(input_ids, token_type_ids, attention_mask)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    micro_f1,macro_f1,average = get_acc_score(pred_labels,true_labels)
    return micro_f1,macro_f1,average
    # acc_score = get_acc_score(true_labels, pred_labels)
    # return acc_score

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument("--resume", action='store_true', default=False,
  #           help="use checkpoint")
  parser.add_argument('--seed', type=int, default=42,
            help="random seed for initialization")
  parser.add_argument('--model_name',type=str,default="casefactor")
  args = parser.parse_args()
  args.model_path = "./model/{}.pkl".format(args.model_name)
  set_seed(args)
  if args.model_name in ["TextCNN", "DPCNN", "TextRNN","TextRNN_Att"]:
      from utils_data import build_dataset, build_iterator
      x = import_module(args.model_name)
      config = x.Config()
      vocab, train_data, dev_data, test_data = build_dataset(config)
      train_dataloader = build_iterator(train_data, config)
      dev_dataloader = build_iterator(dev_data, config)
      test_dataloader = build_iterator(test_data, config)
      config.n_vocab = len(vocab)
      train(args)
  elif args.model_name == "FastText":
      from utils_fasttext import build_dataset, build_iterator
      x = import_module(args.model_name)
      config = x.Config()
      vocab, train_data, dev_data, test_data = build_dataset(config)
      train_dataloader = build_iterator(train_data, config)
      dev_dataloader = build_iterator(dev_data, config)
      test_dataloader = build_iterator(test_data, config)
      config.n_vocab = len(vocab)
      train(args)
  elif args.model_name == "roberta":
      train_path = "./bert_data/train.json"
      dev_path = "./bert_data/dev.json"
      test_path = "./bert_data/test.json"
      label2idx_path = "./bert_data/label2idx.json"
      label2idx = load_json(label2idx_path)
      class_num = len(label2idx)
      train_dataset = RobertaDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
      dev_dataset = RobertaDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
      test_dataset = RobertaDataSet(test_path, max_len=max_len, label2idx_path=label2idx_path)
      test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      train(args)
  else:
      train_path = "./bert_data/train.json"
      dev_path = "./bert_data/dev.json"
      test_path = "./bert_data/test.json"
      label2idx_path = "./bert_data/label2idx.json"
      label2idx = load_json(label2idx_path)
      class_num = len(label2idx)
      train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
      dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
      test_dataset = MultiClsDataSet(test_path, max_len=max_len, label2idx_path=label2idx_path)
      test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      train(args)