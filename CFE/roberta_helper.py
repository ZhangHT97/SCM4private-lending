# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from data_preprocess import load_json


class RobertaDataSet(Dataset):
    def __init__(self, data_path, max_len=128, label2idx_path="./bert_data/label2idx.json"):
        self.label2idx = load_json(label2idx_path)
        self.class_num = len(self.label2idx)
        self.tokenizer = RobertaTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.max_len = max_len
        self.input_ids, self.attention_mask, self.labels = self.encoder(data_path)

    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path,encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                texts.append(line["text"])
                tmp_label = [0] * self.class_num
                for idx,l in enumerate(line["label"]):
                  idc = self.label2idx.get(l,1)
                  tmp_label[idc] = 1
                labels.append(tmp_label)

        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        attention_mask = tokenizers["attention_mask"]
        return input_ids,  attention_mask, torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input_ids[item],self.attention_mask[item], self.labels[item]