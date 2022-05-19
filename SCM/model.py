import json
import logging
import os
import random
from typing import Tuple, List
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from net import TripleMatch
from my_encoder import Encoder

logger = logging.getLogger("train model")
PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'

PAD_IDX = 0
UNK_IDX = 1

class HyperParameters(object):
    """
    用于管理模型超参数
    """

    def __init__(
        self,
        max_length: int = 128,
        epochs=4,
        batch_size=32,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        warmup_steps=0.1,
        embed_dim=300,
        filter_num=200,
        filter_sizes=[2, 3, 4],
        textcnn_dropout=0.3,
        part_num = 5,
        cross_margin = 5,

    ) -> None:
        self.max_length = max_length
        """句子的最大长度"""
        self.epochs = epochs
        """训练迭代轮数"""
        self.batch_size = batch_size
        """每个batch的样本数量"""
        self.learning_rate = learning_rate
        """学习率"""
        self.max_grad_norm = max_grad_norm
        """最大梯度裁剪"""
        self.warmup_steps = warmup_steps
        """学习率线性预热步数"""
        self.embed_dim = embed_dim
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.textcnn_dropout = textcnn_dropout
        self.part_num = part_num
        self.cross_margin = cross_margin

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


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

    if model_class == TripleMatch:
        return two_pair_collate_fn


algorithm_map = {"TripleMatch": TripleMatch}

class ModelTrainer(object):
    def __init__(
        self,
        dataset_path,
        param: HyperParameters,
        algorithm,
        test_input_path,
        test_ground_truth_path,
        vocab_size,
        dic,
    ) -> None:
        """

        :param dataset_path: 数据集路径。 默认当作是训练集，但当train函数采用了kfold参数时，将对该数据集进行划分并做交叉验证
        :param model_dir: 模型路径
        :param param: 超参数
        :param algorithm: 选择算法，默认 MatchModel
        :param test_input_path: 固定的测试集的路径，用于快速测试模型性能
        :param test_ground_truth_path: 固定的测试集的标记
        """
        self.dataset_path = dataset_path
        self.param = param
        self.test_input_path = test_input_path
        self.test_ground_truth_path = test_ground_truth_path
        self.vocab_size = vocab_size
        self.algorithm = algorithm
        self.model_class = algorithm_map[self.algorithm]
        self.dictionary = dic
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info("算法:" + algorithm)

    def load_dataset(
        self, n_splits: int = 1
    ) -> List[Tuple[TripletTextDataset, TripletTextDataset, List[str]]]:
        """
        划分k折交叉验证数据集用于cv

        :param n_splits:
        :return: List[(train_data, test_data, test_labels_list)]
        """

        data = []

        if n_splits == 1:
            train_data = TripletTextDataset.from_jsons(self.dataset_path, use_augment=False)
            test_data = TripletTextDataset.from_jsons(self.test_input_path)
            with open(self.test_ground_truth_path) as f:
                test_label_list = [line.strip() for line in f.readlines()]
            data.append((train_data, test_data, test_label_list))
            return data
        raw_data_list = []
        with open(self.dataset_path, encoding="utf-8") as raw_input:
            for line in raw_input:
                raw_data_list.append(json.loads(line.strip(), encoding="utf-8"))
        with open('data/valid/input_4kf.txt', encoding="utf-8") as raw_test:
            for line_t in raw_test:
                raw_data_list.append(json.loads(line_t.strip(), encoding="utf-8"))
        kf = KFold(n_splits, shuffle=True, random_state=42)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        for train_index, test_index in kf.split(raw_data_list):
            # 准备训练集
            train_data_list = [raw_data_list[i] for i in train_index]
            train_data = TripletTextDataset.from_dict_list(train_data_list, use_augment=False)

            test_data_list = [raw_data_list[i] for i in test_index]
            shuffled_test_data_list = []
            test_label_list = []
            for item in test_data_list:
                a = item["A"]
                b = item["B"]
                c = item["C"]
                label = item["label"]
                item = {"A": a, "B": c, "C": b}
                shuffled_test_data_list.append(item)
                test_label_list.append(label)
            test_data = TripletTextDataset.from_dict_list(shuffled_test_data_list)
            data.append((train_data, test_data, test_label_list))
        return data


    def train(self, model_dir, kfold=1,train=False):
      if train:
        # n_gpu = torch.cuda.device_count()
        n_gpu = 1
        logger.info("***** Running training *****")
        logger.info("dataset: {}".format(self.dataset_path))
        logger.info("k-fold number: {}".format(kfold))
        logger.info("device: {} n_gpu: {}".format(self.device, n_gpu))
        logger.info(
            "config: {}".format(
                json.dumps(self.param.__dict__, indent=4, sort_keys=True)
            )
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        # if n_gpu > 0:
        #     torch.cuda.manual_seed_all(42)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        data = self.load_dataset(kfold)

        all_acc_list = []
        for k, (train_data, test_data, test_label_list) in enumerate(data, start=1):
            one_fold_acc_list = []
            encoder = Encoder(self.param,self.device)
            net = TripleMatch(self.param, self.vocab_size, encoder,self.device).to(self.device)
            net.to(self.device)
            num_train_optimization_steps = (
                int(len(train_data) / self.param.batch_size) * self.param.epochs
            )

            param_optimizer = list(net.named_parameters())
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                  "params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                  "weight_decay": 0.01,
              },
              {
                  "params": [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
                  "weight_decay": 0.0,
              },
            ]

            if self.param.warmup_steps < 1:
                num_warmup_steps = (num_train_optimization_steps * self.param.warmup_steps)
                warm_up_epochs = (self.param.warmup_steps*self.param.epochs)
            else:
                num_warmup_steps = self.param.warmup_steps
                warm_up_epochs = math.ceil(self.param.warmup_steps / (int(len(train_data) / self.param.batch_size)))

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.param.learning_rate, eps=1e-8)
            # warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch <= warm_up_epochs else \
            #    0.5 * (math.cos((epoch - warm_up_epochs) / (self.param.epochs - warm_up_epochs) * math.pi) + 1)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [5,15,20,25,30,40], gamma = 0.001, last_epoch=-1)
            # scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,
            #                                               num_training_steps=num_train_optimization_steps)
            # scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,
            #                                             num_training_steps=num_train_optimization_steps)

            if n_gpu > 1:
                net = torch.nn.DataParallel(net)

            global_step = 0
            net.zero_grad()

            logger.info("***** fold {}/{} *****".format(k, kfold))
            logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.param.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            train_sampler = RandomSampler(train_data)
            collate_fn = get_collator(self.param.max_length, self.device, self.model_class, self.dictionary)
            train_dataloader = DataLoader(
                dataset=train_data,
                sampler=train_sampler,
                batch_size=self.param.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                drop_last=True,
            )
            net.train()
            best_acc = 0
            for epoch in range(int(self.param.epochs)):
                tr_loss = 0
                steps = tqdm(train_dataloader)
                for step, batch in enumerate(steps):
                    """
                    batch(list): tensor0 tensor1 tensor2 tensor3
                    tensor0 [bsz,max-len]   -------> a
                    tensor1 [bsz,max-len]   -------> b
                    tensor2 [bsz,max-len]   -------> c
                    tensor3 [bsz,1]
                    """
                    loss = net(batch,mode="loss")

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.param.max_grad_norm )
                    tr_loss += loss.item()
                    optimizer.step()
                    # scheduler.step()  # Update learning rate schedule
                    net.zero_grad()
                    global_step += 1
                    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                    steps.set_description("Epoch {}/{}, Loss {:.7f},lr{:.10f}".format(epoch + 1, self.param.epochs, loss.item(),learning_rate))
                scheduler.step()
                acc, loss = self.evaluate(net, test_data, test_label_list)
                if acc > best_acc:
                    torch.save(net, os.path.join(model_dir, "model_best"))
                    best_acc = acc
                one_fold_acc_list.append(acc)
                logger.info("Epoch {}, train Loss: {:.7f}, eval acc: {}, eval loss: {:.7f}".format(
                    epoch + 1, tr_loss, acc, loss))
                net.train()
            all_acc_list.append(one_fold_acc_list)
            torch.save(net, os.path.join(model_dir, "model"))

        logger.info("***** Stats *****")
        # 计算kfold的平均的acc
        all_epoch_acc = list(zip(*all_acc_list))
        logger.info("acc for each epoch:")
        for epoch, acc in enumerate(all_epoch_acc, start=1):
            logger.info(
                "epoch %d, mean: %.5f, std: %.5f"
                % (epoch, float(np.mean(acc)), float(np.std(acc)))
            )

        logger.info("***** Training complete *****")
        # test
        net = torch.load(os.path.join(model_dir, "model"))
        test_acc,p,r,f1 = self.test(net, 'data/test/input_clean.txt', 'data/test/ground_truth.txt')
        logger.info("acc for test: %.5f,%.3f,%.3f,%.3f" % (float(test_acc),float(p),float(r),float(f1)))
        net = torch.load(os.path.join(model_dir, "model_best"))
        best_test_acc,best_p,best_r,best_f1 = self.test(net, 'data/test/input_clean.txt', 'data/test/ground_truth.txt')
        logger.info("acc for best_test_acc: %.5f,%.3f,%.3f,%.3f" % (float(best_test_acc),float(best_p),float(best_r),float(best_f1)))
      else:
        # test
        net = torch.load(os.path.join(model_dir, "model"))
        test_acc,p,r,f1 = self.test(net, 'data/test/input_clean.txt', 'data/test/ground_truth.txt')
        logger.info("acc for test: %.5f,%.3f,%.3f,%.3f" % (float(test_acc),float(p),float(r),float(f1)))
        net = torch.load(os.path.join(model_dir, "model_best"))
        best_test_acc,best_p,best_r,best_f1 = self.test(net, 'data/test/input_clean.txt', 'data/test/ground_truth.txt')
        logger.info("acc for best_test_acc: %.5f,%.3f,%.3f,%.3f" % (float(best_test_acc),float(best_p),float(best_r),float(best_f1)))

    def evaluate(self, model, data: TripletTextDataset, real_label_list: List[str]):
        """
        评估模型，计算acc

        :param model:
        :param data:
        :param real_label_list:
        :return:
        """
        sampler = SequentialSampler(data)
        collate_fn = get_collator(self.param.max_length, self.device, self.model_class, self.dictionary)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.param.batch_size,
                                collate_fn=collate_fn, drop_last=True)
        predict_result = []
        loss_sum = 0
        batch_num = 0
        for batch in dataloader:
            with torch.no_grad():
                output = model(batch, mode="evaluate")
                loss = output[1].mean().cpu().item()
                loss_sum += loss
                predict_results = output[0].cpu().numpy()
                cata_indexes = np.argmax(predict_results, axis=1)
                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = "B" if cata_index == 0 else "C"
                    predict_result.append((str(label), float(prob)))
                batch_num +=1

        real_num = batch_num*self.param.batch_size
        real_label_list = real_label_list[:real_num]
        correct = 0
        for i, real_label in enumerate(real_label_list):
            try:
                predict_label = predict_result[i][0]
                if predict_label == real_label:
                    correct += 1
            except Exception as e:
                print(e)
                continue

        acc = correct / len(real_label_list)
        return acc, loss_sum

    def test(self, model, test_input_path, test_ground_truth_path):
        """
        评估模型，计算acc

        :param model:
        :param data:
        :param real_label_list:
        :return:
        """
        test_data = TripletTextDataset.from_jsons(test_input_path)
        with open(test_ground_truth_path) as f:
            label_list = [line.strip() for line in f.readlines()]
        sampler = SequentialSampler(test_data)
        collate_fn = get_collator(self.param.max_length, self.device, self.model_class, self.dictionary)
        dataloader = DataLoader(test_data, sampler=sampler, batch_size=128,collate_fn=collate_fn, drop_last=True)
        predict_result = []
        batch_num = 0
        for batch in dataloader:
            with torch.no_grad():
                output = model(batch, mode="evaluate")
                predict_results = output[0].cpu().numpy()
                cata_indexes = np.argmax(predict_results, axis=1)
                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = "B" if cata_index == 0 else "C"
                    predict_result.append((str(label), float(prob)))
                batch_num +=1

        real_num = batch_num*self.param.batch_size
        real_label_list = label_list[:real_num]
        correct = 0
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i, real_label in enumerate(real_label_list):
            try:
                predict_label = predict_result[i][0]
                if predict_label == real_label:
                    correct += 1
                if predict_label == "B" and real_label =="B":
                    tp +=1
                if predict_label == "B" and real_label =="C":
                    fp += 1
                if predict_label == "C" and real_label =="B":
                    fn += 1
                if predict_label == "C" and real_label =="C":
                    tn += 1
            except Exception as e:
                print(e)
                continue
        acc = correct / len(real_label_list)
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = 2*p*r/(p+r)
        return acc,p,r,f1
