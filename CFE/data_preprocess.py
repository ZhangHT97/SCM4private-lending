# -*- coding: utf-8 -*-

"""
数据预处理
"""

import json
import argparse

def load_json(data_path):
    with open(data_path,encoding="utf-8") as f:
        return json.loads(f.read())


def dump_json(project, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False)


def preprocess(use_word,data_path,max_len_ratio=0.9):
    """

    """
    text_length = []
    if use_word:
        label2idx_path = "./data/label2idx.json"
        f1 = open("./data/train.json", 'r', encoding='utf-8')
        f2 = open("./data/dev.json", 'r', encoding='utf-8')
        f3 = open("./data/test.json", 'r', encoding='utf-8')
        label_lenth = "./data/label_len"
    else:
        train_path = "./"+data_path+"/train.json"
        dev_path = "./"+data_path+"/dev.json"
        test_path = "./"+data_path+"/test.json"
        label2idx_path = "./"+data_path+"/label2idx.json"
        attr_list = "./"+data_path+"/attrlist"
        label_lenth = "./"+data_path+"/label_len"
        f1 = open(train_path, 'r', encoding='utf-8')
        f2 = open(dev_path, 'r', encoding='utf-8')
        f3 = open(test_path, 'r', encoding='utf-8')
    t1 = f1.readlines()
    t2 = f2.readlines()
    t3 = f3.readlines()
    t = t1 + t2 + t3
    att1_label = []
    att2_label = []
    att3_label = []
    att4_label = []
    att5_label = []
    att6_label = []
    att7_label = []
    att8_label = []
    att9_label = []
    att10_label = []
    for data in t:
        data = json.loads(data)
        att1_label.append(data["label"][0])
        att2_label.append(data["label"][1])
        att3_label.append(data["label"][2])
        att4_label.append(data["label"][3])
        att5_label.append(data["label"][4])
        att6_label.append(data["label"][5])
        att7_label.append(data["label"][6])
        att8_label.append(data["label"][7])
        att9_label.append(data["label"][8])
        att10_label.append(data["label"][9])
        if use_word:
            text_length.append(len(data["text"].split()))
        else:
            text_length.append(len(data["text"]))
        # labels.extend(data["label"])
    # with open(train_data_path,encoding="utf-8") as f:
    #     for data in f:
    #         data = json.loads(data)
    #         if use_word:
    #             text_length.append(len(data["text"].split()))
    #         else:
    #             text_length.append(len(data["text"]))
    #         labels.extend(data["label"])

    label_len = []
    att1_label = list(set(att1_label))
    label_len.append(len(att1_label))

    att2_label = list(set(att2_label))
    label_len.append(len(att2_label))

    att3_label = list(set(att3_label))
    label_len.append(len(att3_label))

    att4_label = list(set(att4_label))
    label_len.append(len(att4_label))

    att5_label = list(set(att5_label))
    label_len.append(len(att5_label))

    att6_label = list(set(att6_label))
    label_len.append(len(att6_label))

    att7_label = list(set(att7_label))
    label_len.append(len(att7_label))

    att8_label = list(set(att8_label))
    label_len.append(len(att8_label))

    att9_label = list(set(att9_label))
    label_len.append(len(att9_label))

    att10_label = list(set(att10_label))
    label_len.append(len(att10_label))

    att_label ={"借款交付形式":att1_label,"借款人基本属性":att2_label,"借款用途":att3_label,"借贷合意的凭据":att4_label,
                "出借人基本属性":att5_label,"出借意图":att6_label,"担保类型":att7_label,"约定期内利率":att8_label,
                "约定计息方式":att9_label,"还款交付形式":att10_label}
    labels=att1_label+att2_label+att3_label+att4_label+att5_label+att6_label+att7_label+att8_label+att9_label+att10_label
    label2idx = {label: idx for idx, label in enumerate(labels)}
    dump_json(label2idx, label2idx_path)
    if data_path == "bertlabel_data":
        dump_json(att_label, attr_list)
    dump_json(label_len, label_lenth)

    text_length.sort()

    print("当设置max_len={}时，可覆盖{}的文本".format(text_length[int(len(text_length)*max_len_ratio)], max_len_ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_word", action='store_true', default=False,
              help="use split word data")
    parser.add_argument("--data_path", type=str, default="bertlabel_data",
                        help="data path")
    args = parser.parse_args()
    preprocess(args.use_word,args.data_path)
