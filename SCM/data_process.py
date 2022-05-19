import json
import re
# from pyltp import Segmentor
import jieba
import os

# def word_splitter(sentence):  # 分词
#     segmentor = Segmentor()  # 初始化实例
#     segmentor.load('./LTP/cws.model')  # 加载模型
#     words = segmentor.segment(sentence)  # 分词
#     words_list = list(words)
#     segmentor.release()  # 释放模型
#     return words_list
PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'

PAD_IDX = 0
UNK_IDX = 1

def stopwordslist(filepath):    # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()] # 以行的形式读取停用词表，同时转换为列表
    return stopword

def data_process(data_path,test = None):
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stopwords = stopwordslist('./stopwords.txt')
            jieba.load_userdict('./userdict.txt')
            x = json.loads(line)
            pattern = re.compile("某+(\d)")
            pattern0 = re.compile("(xx)|(x)|(X)|(XX)")
            # pattern1 = re.compile("\d+[:：；,，、。．.]")
            pattern2 = re.compile("(一|二|三|四|五|六|七|八|九|十)+[:：；,，、。．.]")
            pattern3 = re.compile(
                "(\d{12})|(\d{13})|(\d{14})|(\d{15})|(\d{16}|(\d{17})|(\d{18})|(\d{19})|(\d{20}))")
            BankCard_pattern = re.compile("^([1-9]{1})(\d{14}|\d{18})$")
            Tell_pattern = re.compile("^(13[0-9]|14[5|7]|15[0|1|2|3|4|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}$")
            IDCards_pattern = re.compile("(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)")
            # Date_pattern = re.compile("\d{4}年\d{1,2}月\d{1,2}日|\d{4}年")
            Last_pattern = re.compile("[a-zA-Z\"\"{}＊:：；,，、。．.￥（）《》— -'\\']")

            data = x["A"]
            tmp_data = re.sub(pattern, '某', data)
            tmp_data = re.sub(pattern0, '某', tmp_data)
            # tmp_data = re.sub(pattern1, '', tmp_data)
            tmp_data = re.sub(pattern2, '', tmp_data)
            tmp_data = re.sub(pattern3, '', tmp_data)
            tmp_data = re.sub(BankCard_pattern, '', tmp_data)
            tmp_data = re.sub(Tell_pattern, '', tmp_data)
            tmp_data = re.sub(IDCards_pattern, '', tmp_data)
            # tmp_data = re.sub(Date_pattern, '某年某月某日', tmp_data)
            tmp_data = re.sub(Last_pattern, '', tmp_data)
            tmp_data = jieba.lcut(tmp_data, cut_all=False)  # 精确模式 default
            x["A"] = [word for word in tmp_data if word not in stopwords and word!="\n"]  # 去除停用词  list
            x["A"] = list(filter(None, x["A"]))  # 过滤空字符和None

            data = x["B"]
            tmp_data = re.sub(pattern, '某', data)
            tmp_data = re.sub(pattern0, '某', tmp_data)
            # tmp_data = re.sub(pattern1, '', tmp_data)
            tmp_data = re.sub(pattern2, '', tmp_data)
            tmp_data = re.sub(pattern3, '', tmp_data)
            tmp_data = re.sub(BankCard_pattern, '', tmp_data)
            tmp_data = re.sub(Tell_pattern, '', tmp_data)
            tmp_data = re.sub(IDCards_pattern, '', tmp_data)
            # tmp_data = re.sub(Date_pattern, '某年某月某日', tmp_data)
            tmp_data = re.sub(Last_pattern, '', tmp_data)
            tmp_data = jieba.lcut(tmp_data, cut_all=False)  # 精确模式 default
            x["B"] = [word for word in tmp_data if word not in stopwords and word!="\n" ]  # 去除停用词  list
            x["B"] = list(filter(None, x["B"]))  # 过滤空字符和None

            data = x["C"]
            tmp_data = re.sub(pattern, '某', data)
            tmp_data = re.sub(pattern0, '某', tmp_data)
            # tmp_data = re.sub(pattern1, '', tmp_data)
            tmp_data = re.sub(pattern2, '', tmp_data)
            tmp_data = re.sub(pattern3, '', tmp_data)
            tmp_data = re.sub(BankCard_pattern, '', tmp_data)
            tmp_data = re.sub(Tell_pattern, '', tmp_data)
            tmp_data = re.sub(IDCards_pattern, '', tmp_data)
            # tmp_data = re.sub(Date_pattern, '某年某月某日', tmp_data)
            tmp_data = re.sub(Last_pattern, '', tmp_data)
            tmp_data = jieba.lcut(tmp_data, cut_all=False)  # 精确模式 default
            x["C"] = [word for word in tmp_data if word not in stopwords and word!="\n"]  # 去除停用词  list
            x["C"] = list(filter(None, x["C"]))  # 过滤空字符和None

            if test:
                data_dic = {"A": x["A"], "B": x["B"], "C": x["C"]}
            else:
                data_dic = {"A": x["A"], "B": x["B"], "C": x["C"], "label": x["label"]}
            json_str = json.dumps(data_dic, ensure_ascii=False)
            data_clean.write(json_str + '\n')

def create_dict(data1_path,data2_path,data3_path,dict_path):
    tmp_dict = {}
    with open(data1_path, "r", encoding="utf-8") as f1:
        for line in f1:
            x = json.loads(line)
            for word in x["A"]:
                num = tmp_dict.get(word,0)+1
                tmp_dict[word] = num
            for word in x["B"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num
            for word in x["C"]:
                num = tmp_dict.get(word,0)+1
                tmp_dict[word] = num

    with open(data2_path, "r", encoding="utf-8") as f2:
        for line in f2:
            x = json.loads(line)
            for word in x["A"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num
            for word in x["B"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num
            for word in x["C"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num

    with open(data3_path, "r", encoding="utf-8") as f3:
        for line in f3:
            x = json.loads(line)
            for word in x["A"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num
            for word in x["B"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num
            for word in x["C"]:
                num = tmp_dict.get(word, 0) + 1
                tmp_dict[word] = num

        tmp_dict = sorted(tmp_dict.items(),key=lambda d:d[1],reverse=True)
        our_dict = {}
        our_dict[PAD_WORD] = PAD_IDX
        our_dict[UNK_WORD] = UNK_IDX
        idx = 2
        for item in tmp_dict:
            if item[1] >= 2:
                our_dict[item[0]] = idx
                idx += 1
        if not os.path.exists(dict_path):
            with open("data/dict",'w',encoding='utf-8') as f:
                for i in our_dict:
                    f.write(str(i)+'\n')
        return our_dict

if __name__ == '__main__':
    # data_clean = open('data/train/input_clean.txt', 'w+', encoding='utf-8')
    # data_process('data/train/input.txt',test = False)
    # print("已完成训练数据预处理")
    # data_clean = open('data/valid/input_clean.txt', 'w+', encoding='utf-8')
    # data_process('data/valid/input.txt',test = True)
    # print("已完成验证数据预处理")
    # data_clean = open('data/test/input_clean.txt', 'w+', encoding='utf-8')
    # data_process('data/test/input.txt', test=True)
    # print("已完成测试数据预处理")
    dictionary = create_dict("data/train/input_clean.txt","data/test/input_clean.txt",
                             'data/test/input_clean.txt',"data/dict")
    exit(0)
