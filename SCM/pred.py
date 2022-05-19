import torch
import jieba
import json
import re
PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'

PAD_IDX = 0
UNK_IDX = 1
max_len = 399
dict_path = "data/dict"
jieba.load_userdict('./userdict.txt')

with open(dict_path, "r", encoding='utf-8') as f:
    dic = {}
    for i, data in enumerate(f.readlines()):
        word = data.strip('\n')
        dic[word] = i

stopwords = [line.strip() for line in open('./stopwords.txt', 'r',encoding='utf-8').readlines()]

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model/model", map_location=device)

def data_pre(data):
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
    x = [word for word in tmp_data if word not in stopwords and word != "\n"]  # 去除停用词  list
    x = list(filter(None, x))  # 过滤空字符和None
    return x


def data_embedding(text_a,text_b):
    texta_idx = []
    textb_idx = []
    for word1 in text_a:
        if word1 != "\n":
            texta_idx.append(dic.get(word1, UNK_IDX))
    for word2 in text_b:
        if word2 != "\n":
            textb_idx.append(dic.get(word2, UNK_IDX))

    lenth_a = (len(texta_idx))
    lenth_b = (len(textb_idx))

    if len(texta_idx) < 399:
        padding_a = [0] * (max_len - len(texta_idx))
        texta_idx += padding_a
    if len(textb_idx) < max_len:
        padding_b = [0] * (max_len - len(textb_idx))
        textb_idx += padding_b

    a_tensor = torch.LongTensor(texta_idx).unsqueeze(0).to(device)
    b_tensor = torch.LongTensor(textb_idx).unsqueeze(0).to(device)
    c_tensor = a_tensor
    label_tensor = torch.LongTensor([0]).to(device)
    lenth_tensor = torch.LongTensor((lenth_a, lenth_b, lenth_b)).unsqueeze(0).to(device)
    example_tensors = (a_tensor, b_tensor, c_tensor, label_tensor, lenth_tensor)
    return example_tensors

if __name__ == '__main__':
    str1 = input("请输入A：")
    str2 = input("请输入B：")
    a = data_pre(str1)
    b = data_pre(str2)
    inp = data_embedding(a,b)
    score1,score2 = model(inp,mode="use_model")
    score1 = score1.tolist()[0]
    score2 = score2.tolist()[0]
    if score1 > score2:
        out = score2 / score1
    else:
        out = score1 / score2
    print(out)