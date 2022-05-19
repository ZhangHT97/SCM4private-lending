import json
import matplotlib.pyplot as plt
import re
import numpy as np
import jieba
import argparse
def data4bert(data_path,train_path,dev_path,test_path):
    f = open(data_path, encoding='utf-8')
    t = json.load(f)
    data = []
    pattern = re.compile('原告.+\n\n')
    for i,doc in enumerate(t):
        label = []
        value = list(doc["attr"].values())
        for idx in range(len(value)):
            value[idx] = value[idx][0]  # 选取对应attr下的第一个标签
        if value[0] == "其他":
            value[0] = "其他交付形式"
        elif value[0] == "未知或模糊":
            value[0] = "未知借款交付形式"
        if value[1] == "自然人":
            value[1] = "借款自然人"
        elif value[1] == "法人":
            value[1] = "借款法人"
        elif value[1] == "其他组织":
            value[1] = "借款其他组织"
        if value[2] == "其他":
            value[2] = "其他借款意图"
        if value[3] == "其他":
            value[3] = "其他凭据"
        elif value[3] == "未知或模糊":
            value[3] = "未说明凭据情况"
        if value[4] == "自然人":
            value[4] = "还款自然人"
        elif value[4] == "法人":
            value[4] = "还款法人"
        elif value[4] == "其他组织":
            value[4] = "还款其他组织"
        if value[5] == "其他":
            value[5] = "其他出借意图"
        if value[7] == "其他":
            value[7] = "不清楚利率情况"
        if value[8] == "其他":
            value[8] = "其他计息方式"
        if value[9] == "其他":
            value[9] = "其他还款方式"
        elif value[9] == "未知或模糊":
            value[9] = "未说明还款方式"
        elif value[9] == "现金":
            value[9] = "现金还款"
        elif value[9] == "银行转账":
            value[9] = "转账还款"
        elif value[9] == "票据":
            value[9] = "票据还款"
        elif value[9] == "网上电子汇款":
            value[9] = "网上电子还款"
        label.append(value[0])
        label.append(value[1])
        label.append(value[2])
        label.append(value[3])
        label.append(value[4])
        label.append(value[5])
        label.append(value[6])
        label.append(value[7])
        label.append(value[8])
        label.append(value[9])
        # new_data = json.dumps({"fact": doc["fact"], "label": label}, ensure_ascii=False).strip()
        # new.write(new_data + "\n")
        fact = doc["fact"]
        fact = re.sub(pattern, "", fact, count=1)  # 提出基本情况部分
        fact = fact.replace("\n", "").replace(" ", "").replace("，", "")\
            .replace("。", "").replace("：", "").replace("“", "").replace("”", "").replace("、", "")
        # value = list(doc["attr"].values())
        new_data = {"text":fact,"label":label}
        data.append(new_data)
    # data
    f.close()
    train_num = 0
    dev_num = 0
    test_num = 0
    with open(train_path, 'w', encoding='utf-8') as f1,\
            open(dev_path, 'w', encoding='utf-8') as f2,\
            open(test_path, 'w', encoding='utf-8') as f3:
        for num,dic in enumerate(data):
            js = json.dumps(dic,ensure_ascii=False)
            if num <= 2525:
                train_num += 1
                f1.write(js+"\n")
            elif 2525 <= num <= 2840:
                dev_num += 1
                f2.write(js+"\n")
            elif 2841 <= num <= 3155:
                test_num += 1
                f3.write(js+"\n")
    print("训练样本：{}个，验证样本：{}个，测试样本：{}个".format(train_num,dev_num,test_num))

def lenth_count(raw):
    f = open(raw, encoding='utf-8')
    t = json.load(f)
    lenth = []
    pattern = re.compile('原告.+\n\n')
    for i, doc in enumerate(t):
        fact = doc["fact"]
        fact = re.sub(pattern, "", fact, count=1)  # 提出基本情况部分
        fact = fact.replace("\n", "").replace(" ", "").replace("，", "")\
            .replace("。", "").replace("：", "").replace("“","").replace("”","").replace("、","")
        tmp_len = len(fact)
        lenth.append(tmp_len)
    plt.hist(lenth, bins=15, density=False, facecolor="blue", edgecolor="black", range=(400,900))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    font = {'family': 'SimHei',
            'weight': 'normal',
            'size': 20,
            }
    # 显示横轴标签
    plt.xlabel("长度",fontdict=font)
    # 显示纵轴标签
    plt.ylabel("频数",fontdict=font)
    # 显示图标题
    plt.title("事实描述文书长度统计(全文)",fontdict=font)
    plt.show()
    lenth_np = np.array(lenth)
    len_mean = lenth_np.mean()
    len_max = lenth_np.max()
    len_min = lenth_np.min()
    print("文本平均长：{}，最长：{}，最短：{}".format(len_mean,len_max,len_min))

def stopwordslist(filepath):    # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()] # 以行的形式读取停用词表，同时转换为列表
    return stopword

def data_splitword(data_path,train_path,dev_path,test_path):
    f = open(data_path, encoding='utf-8')
    t = json.load(f)
    train_num = 0
    dev_num = 0
    test_num = 0
    pattern = re.compile('原告.+\n\n')
    stopwords = stopwordslist('./stopwords.txt')
    # jieba.load_userdict('./userdict.txt')
    with open(train_path, 'w', encoding='utf-8') as f1, \
            open(dev_path, 'w', encoding='utf-8') as f2, \
            open(test_path, 'w', encoding='utf-8') as f3:
        for i, doc in enumerate(t):
            str_fact =""
            fact = doc["fact"]
            fact = re.sub(pattern, "", fact, count=1)  # 提出基本情况部分
            fact = fact.replace("\n", "").replace(" ", "").replace("，", "") \
                .replace("。", "").replace("：", "").replace("“", "").replace("”", "").replace("、", "")
            tmp_fact = jieba.lcut(fact, cut_all=False)  # 精确模式 default
            doc["fact"] = [word for word in tmp_fact if word not in stopwords and word!="\n"]  # 去除停用词  list
            doc["fact"] = list(filter(None, doc["fact"]))  # 过滤空字符和None
            for string in doc["fact"]:
                tmp_str = str(string)+" "
                str_fact += tmp_str
            label = []
            value = list(doc["attr"].values())
            for idx in range(len(value)):
                value[idx] = value[idx][0]  # 选取对应attr下的第一个标签
            if value[0] == "其他":
                value[0] = "其他交付形式"
            elif value[0] == "未知或模糊":
                value[0] = "未知借款交付形式"
            if value[1] == "自然人":
                value[1] = "借款自然人"
            elif value[1] == "法人":
                value[1] = "借款法人"
            elif value[1] == "其他组织":
                value[1] = "借款其他组织"
            if value[2] == "其他":
                value[2] = "其他借款意图"
            if value[3] == "其他":
                value[3] = "其他凭据"
            elif value[3] == "未知或模糊":
                value[3] = "未说明凭据情况"
            if value[4] == "自然人":
                value[4] = "还款自然人"
            elif value[4] == "法人":
                value[4] = "还款法人"
            elif value[4] == "其他组织":
                value[4] = "还款其他组织"
            if value[5] == "其他":
                value[5] = "其他出借意图"
            if value[7] == "其他":
                value[7] = "不清楚利率情况"
            if value[8] == "其他":
                value[8] = "其他计息方式"
            if value[9] == "其他":
                value[9] = "其他还款方式"
            elif value[9] == "未知或模糊":
                value[9] = "未说明还款方式"
            elif value[9] == "现金":
                value[9] = "现金还款"
            elif value[9] == "银行转账":
                value[9] = "转账还款"
            elif value[9] == "票据":
                value[9] = "票据还款"
            elif value[9] == "网上电子汇款":
                value[9] = "网上电子还款"
            label.append(value[0])
            label.append(value[1])
            label.append(value[2])
            label.append(value[3])
            label.append(value[4])
            label.append(value[5])
            label.append(value[6])
            label.append(value[7])
            label.append(value[8])
            label.append(value[9])
            data_dic = {'text':str_fact, 'label': label}
            json_str = json.dumps(data_dic, ensure_ascii=False)
            if i <= 2525:
                train_num += 1
                f1.write(json_str + "\n")
            elif 2525 <= i <= 2840:
                dev_num += 1
                f2.write(json_str + "\n")
            elif 2841 <= i <= 3155:
                test_num += 1
                f3.write(json_str + "\n")
    print("训练样本：{}个，验证样本：{}个，测试样本：{}个".format(train_num, dev_num, test_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_obj', type=str, default="get_data")
    args = parser.parse_args()
    if args.process_obj == "get_data":
        data4bert("./raw_data.json","./bert_data/train.json","./bert_data/dev.json","./bert_data/test.json")
        data4bert("./raw_data.json","./bertlabel_data/train.json","./bertlabel_data/dev.json","./bertlabel_data/test.json")
        data_splitword("./raw_data.json",'./data/train.json','./data/dev.json','./data/test.json')
    if args.process_obj == "len_conut":
        lenth_count("./raw_data.json")
