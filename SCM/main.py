import json
import logging
import sys
import time
import os
import torch
from net import TripleMatch
from torch.utils.data.dataloader import default_collate
from typing import Tuple, List, Union
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from argparse import ArgumentParser


class TripletTextDataset(Dataset):
    def __init__(self, text_a_list, text_b_list, text_c_list, label_list=None):
        if label_list is None or len(label_list) == 0:
            label_list = [None] * len(text_a_list)
        assert all(
            len(label_list) == len(text_list)
            for text_list in [text_a_list, text_b_list, text_c_list]
        )
        self.text_a_list = text_a_list
        self.text_b_list = text_b_list
        self.text_c_list = text_c_list
        self.label_list = [0 if label == "B" else 1 for label in label_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text_a, text_b, text_c, label = (
            self.text_a_list[index],
            self.text_b_list[index],
            self.text_c_list[index],
            self.label_list[index],
        )
        return text_a, text_b, text_c, label

    @classmethod
    def from_dataframe(cls, df):
        text_a_list = df["A"].tolist()
        text_b_list = df["B"].tolist()
        text_c_list = df["C"].tolist()
        if "label" not in df:
            df["label"] = "B"
        label_list = df["label"].tolist()
        return cls(text_a_list, text_b_list, text_c_list, label_list)

    @classmethod
    def from_dict_list(cls, data, use_augment=False):
        df = pd.DataFrame(data)
        if "label" not in df:
            df["label"] = "B"
        if use_augment:
            df = TripletTextDataset.augment(df)
        return cls.from_dataframe(df)

    @classmethod
    def from_jsons(cls, json_lines_file, use_augment=False):
        with open(json_lines_file, encoding="utf-8") as f:
            data = list(map(lambda line: json.loads(line), f))
        return cls.from_dict_list(data, use_augment)

    @staticmethod
    def augment(df):
        df_cp1 = df.copy()
        df_cp1["B"] = df["C"]
        df_cp1["C"] = df["B"]
        df_cp1["label"] = "C"

        df_cp2 = df.copy()
        df_cp2["A"] = df["B"]
        df_cp2["B"] = df["A"]
        df_cp2["label"] = "B"

        df_cp3 = df.copy()
        df_cp3["A"] = df["B"]
        df_cp3["B"] = df["C"]
        df_cp3["C"] = df["A"]
        df_cp3["label"] = "C"

        df_cp4 = df.copy()
        df_cp4["A"] = df["C"]
        df_cp4["B"] = df["A"]
        df_cp4["C"] = df["C"]
        df_cp4["label"] = "C"

        df_cp5 = df.copy()
        df_cp5["A"] = df["C"]
        df_cp5["B"] = df["C"]
        df_cp5["C"] = df["A"]
        df_cp5["label"] = "B"

        df = pd.concat([df, df_cp1, df_cp2, df_cp3, df_cp4, df_cp5])
        df.index = range(len(df))
        tmp_df = df.copy()
        tmp_df['A'] = df['A'].apply(str)
        tmp_df['B'] = df['B'].apply(str)
        tmp_df['C'] = df['C'].apply(str)
        tmp_df['label'] = tmp_df['label'].apply(str)
        tmp_df = tmp_df.drop_duplicates()
        new_df = df.loc[tmp_df.index]
        new_df.index = range(len(new_df))

        new_df = new_df.sample(frac=1)

        return new_df

def get_collator(max_len, device, model_class, dic):
    UNK_IDX = 1
    def two_pair_collate_fn(batch):
        """
        获取一个mini batch的数据，将文本三元组转化成tensor。

        :param batch:
        :return:
        """
        example_tensors = []
        for text_a, text_b, text_c, label in batch:
            texta_idx = []
            textb_idx = []
            textc_idx = []
            for word1 in text_a:
                if word1 !="\n":
                    texta_idx.append(dic.get(word1, UNK_IDX))
            for word2 in text_b:
                if word2 !="\n":
                    textb_idx.append(dic.get(word2, UNK_IDX))
            for word3 in text_c:
                if word3 !="\n":
                    textc_idx.append(dic.get(word3, UNK_IDX))
            lenth_a = (len(texta_idx))
            lenth_b = (len(textb_idx))
            lenth_c = (len(textc_idx))
            if len(texta_idx) < max_len:
                padding_a = [0] * (max_len - len(texta_idx))
                texta_idx += padding_a
            if len(textb_idx) < max_len:
                padding_b = [0] * (max_len - len(textb_idx))
                textb_idx += padding_b
            if len(textc_idx) < max_len:
                padding_c = [0] * (max_len - len(textc_idx))
                textc_idx += padding_c

            a_tensor = torch.LongTensor(texta_idx).to(device)
            b_tensor = torch.LongTensor(textb_idx).to(device)
            c_tensor = torch.LongTensor(textc_idx).to(device)
            label_tensor = torch.LongTensor([label]).to(device)
            lenth_tensor = torch.LongTensor((lenth_a, lenth_b, lenth_c))
            example_tensors.append((a_tensor, b_tensor, c_tensor, label_tensor, lenth_tensor))

        return default_collate(example_tensors)

    if model_class == "TripleMatch":
        return two_pair_collate_fn

def predict(cfg,text_tuples: Union[List[Tuple[str, str, str]], TripletTextDataset],dict) -> List[
    Tuple[str, float]]:
    if isinstance(text_tuples, Dataset):
        data = text_tuples
    else:
        text_a_list, text_b_list, text_c_list = [list(i) for i in zip(*text_tuples)]
        data = TripletTextDataset(text_a_list, text_b_list, text_c_list, None)
    sampler = SequentialSampler(data)
    collate_fn = get_collator(cfg.max_len, cfg.device, "TripleMatch", dict)
    dataloader = DataLoader(data, sampler=sampler, batch_size=16, collate_fn=collate_fn)
    final_results = []
    for batch in dataloader:
        with torch.no_grad():
            predict_results = model(batch, mode="evaluate")[0].numpy()
            cata_indexes = np.argmax(predict_results, axis=1)

            for i_sample, cata_index in enumerate(cata_indexes):
                label = "B" if cata_index == 0 else "C"
                final_results.append(str(label))

    return final_results

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--max_len", type=int, default=399, help="max_length")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    cfg = parser.parse_args()

    return cfg

if __name__ == '__main__':
    logging.disable(sys.maxsize)
    start_time = time.time()
    config = get_args()
    input_path = "data/test/input.txt"
    output_path = "data/output/output.txt"
    dict_path = "data/dict"
    if not os.path.exists("data/output"):
        os.mkdir("data/output")
    inf = open(input_path, "r", encoding="utf-8")
    ouf = open(output_path, "w", encoding="utf-8")
    with open(dict_path, "r", encoding='utf-8') as f:
        dic = {}
        for i, data in enumerate(f.readlines()):
            word = data.strip('\n')
            dic[word] = i
    MODEL_DIR = "model"
    model = TripleMatch.load(MODEL_DIR, version="best")
    # model.load_state_dict(torch.load(MODEL_DIR))

    text_tuple_list = []
    for line in inf:
        line = line.strip()
        items = json.loads(line)
        a = items["A"]
        b = items["B"]
        c = items["C"]
        text_tuple_list.append((a, b, c))

    results = predict(config,text_tuple_list, dic)
    for label in results:
        print(str(label), file=ouf)

    inf.close()
    ouf.close()

    end_time = time.time()
    spent = end_time - start_time
    print("numbers of samples: %d" % len(results))
    print("time spent: %.2f seconds" % spent)
