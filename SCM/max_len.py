import json
import numpy as np
from matplotlib import pyplot as plt


def find_max(train_file,test_file):
    lines=[]
    with open(train_file, encoding="utf-8") as train:
        with open(test_file, encoding="utf-8") as test:
            for line_t in train:
                lines.append(line_t.strip())
            for line_v in test:
                lines.append(line_v.strip())
    docA_max_len = 0
    docB_max_len = 0
    docC_max_len = 0
    docA_min_len = 1000
    docB_min_len = 1000
    docC_min_len = 1000
    for line in lines:
        x = json.loads(line)
        docA_max_len = len(x["A"]) if len(x["A"]) > docA_max_len else docA_max_len
        docA_min_len = len(x["A"]) if len(x["A"]) < docA_min_len else docA_min_len

        docB_max_len = len(x["B"]) if len(x["B"]) > docB_max_len else docB_max_len
        docB_min_len = len(x["B"]) if len(x["B"]) < docB_min_len else docB_min_len

        docC_max_len = len(x["C"]) if len(x["C"]) > docC_max_len else docC_max_len
        docC_min_len = len(x["C"]) if len(x["C"]) < docC_min_len else docC_min_len

    print("A句子最长为：%d句", docA_max_len)
    print("B句子最长为：%d句", docB_max_len)
    print("C句子最长为：%d句", docC_max_len)
    print("****************************")
    print("A句子最短为：%d句", docA_min_len)
    print("B句子最短为：%d句", docB_min_len)
    print("C句子最短为：%d句", docC_min_len)
    return min(docA_min_len,docB_min_len,docC_min_len),max(docA_max_len,docB_max_len,docC_max_len)

def count_num(min_,max_,train,test):
    lenth={}
    for i in range(min_-2,max_+3):
        lenth[i+1] = 0
    lines = []
    with open(train, encoding="utf-8") as f1:
        with open(test, encoding="utf-8") as f2:
            for line_t in f1:
                lines.append(line_t.strip())
            for line_v in f2:
                lines.append(line_v.strip())
    for line in lines:
        x = json.loads(line)
        if len(x["A"]) in lenth:
            lenth[len(x["A"])] += 1
        if len(x["B"]) in lenth:
            lenth[len(x["B"])] += 1
        if len(x["C"]) in lenth:
            lenth[len(x["C"])] += 1

    """避免找不到字体"""
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    plt.rcParams['font.size'] = 12  # 字体大小
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    x = lenth.keys()
    y = lenth.values()
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='#9999ff', width=0.5)
    for lenth,freq in enumerate(y):
        plt.text(lenth+1, freq+1, '%s' % freq,fontsize=10, ha='center', va='bottom')

    plt.title("句子长度统计")
    plt.xlabel('长度')
    plt.ylabel('频数')
    plt.show()

if __name__ == '__main__':
    train_file = 'data/train/input_clean.txt'
    test_file = 'data/test/input_clean.txt'
    # data_path = '../data/valid/valid_clean.json'
    a,b=find_max(train_file,test_file)
    count_num(min_=a,max_=b,train=train_file,test=test_file)