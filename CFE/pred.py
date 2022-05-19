from importlib import import_module
import jieba
import torch
import argparse
import pickle as pkl
import json

PAD = '<PAD>'
UNK = '<UNK>'

PAD_IDX = 0
UNK_IDX = 1

def load_json(data_path):
    with open(data_path,encoding="utf-8") as f:
        return json.loads(f.read())

def pred(args,inp,vocab):
    x = import_module(args.model_name)
    config = x.Config()
    config.n_vocab = len(vocab)
    model = x.Model(config)
    model.eval()
    model.load_state_dict(torch.load('./model/TextCNN.pkl'))
    logits = torch.sigmoid(model(inp))
    # logits = torch.sigmoid(logits).cpu().tolist()
    seq_idx1 = label_lenth[0]
    seq_idx2 = seq_idx1+label_lenth[1]
    seq_idx3 = seq_idx2+label_lenth[2]
    seq_idx4 = seq_idx3+label_lenth[3]
    seq_idx5 = seq_idx4+label_lenth[4]
    seq_idx6 = seq_idx5+label_lenth[5]
    seq_idx7 = seq_idx6+label_lenth[6]
    seq_idx8 = seq_idx7+label_lenth[7]
    seq_idx9 = seq_idx8+label_lenth[8]
    seq_idx10 = seq_idx9+label_lenth[9]

    out1 = logits[:,:seq_idx1]
    out2 = logits[:,seq_idx1:seq_idx2]
    out3 = logits[:,seq_idx2:seq_idx3]
    out4 = logits[:,seq_idx3:seq_idx4]
    out5 = logits[:,seq_idx4:seq_idx5]
    out6 = logits[:,seq_idx5:seq_idx6]
    out7 = logits[:,seq_idx6:seq_idx7]
    out8 = logits[:,seq_idx7:seq_idx8]
    out9 = logits[:,seq_idx8:seq_idx9]
    out10 = logits[:,seq_idx9:seq_idx10]

    attr1_label_id = torch.argmax(out1).cpu().tolist()
    attr2_label_id = (torch.argmax(out2)+seq_idx1).cpu().tolist()
    attr3_label_id = (torch.argmax(out3)+seq_idx2).cpu().tolist()
    attr4_label_id = (torch.argmax(out4)+seq_idx3).cpu().tolist()
    attr5_label_id = (torch.argmax(out5)+seq_idx4).cpu().tolist()
    attr6_label_id = (torch.argmax(out6)+seq_idx5).cpu().tolist()
    attr7_label_id = (torch.argmax(out7)+seq_idx6).cpu().tolist()
    attr8_label_id = (torch.argmax(out8)+seq_idx7).cpu().tolist()
    attr9_label_id = (torch.argmax(out9)+seq_idx8).cpu().tolist()
    attr10_label_id = (torch.argmax(out10)+seq_idx9).cpu().tolist()
    label_ids = [attr1_label_id,attr2_label_id,attr3_label_id,attr4_label_id,attr5_label_id,attr6_label_id,
                 attr7_label_id,attr8_label_id,attr9_label_id,attr10_label_id]
    pred_label = []
    for id,v in enumerate(label_ids):
        pred_label.append(idx2label[v])
    json_out = {"借款交付形式": pred_label[0], "借款人基本属性": pred_label[1], "借款用途": pred_label[2], "借贷合意的凭据": pred_label[3],
         "出借人基本属性": pred_label[4], "出借意图": pred_label[5], "担保类型": pred_label[6], "约定期内利率": pred_label[7],
         "约定计息方式": pred_label[8], "还款交付形式": pred_label[9]}
    return json_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="TextCNN")
    parser.add_argument('--data_path', type=str, default="./data/vocab.txt")
    parser.add_argument('--pad_size', type=int, default=300)
    args = parser.parse_args()
    vocab = pkl.load(open(args.data_path, 'rb'))
    label2idx = load_json("./data/label2idx.json")
    idx2label = {}
    for key, value in label2idx.items():
        idx2label[value] = key
    label_lenth = load_json("./data/label_len")
    inp = input("请输入裁判文书事实描述：")
    fact = inp.replace("\\n", "").replace(" ", "").replace("，", "") \
        .replace("。", "").replace("：", "").replace("“", "").replace("”", "").replace("、", "")
    fact = jieba.lcut(fact, cut_all=False)
    fact = list(filter(None, fact))  # 过滤空字符和None
    words_line = []
    seq_len = len(fact)
    if args.pad_size:
        if len(fact) < args.pad_size:
            fact.extend([PAD] * (args.pad_size - len(fact)))
        else:
            fact = fact[:args.pad_size]
            seq_len = args.pad_size
    # word to id
    for word in fact:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    inp_tensor = torch.tensor(words_line).unsqueeze(0)
    model_inp = (inp_tensor,None)
    out = pred(args, model_inp,vocab)
    print(out)
