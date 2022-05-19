import torch
import argparse
import json
from transformers import BertTokenizer
from BERT_Label import Bert_label
PAD = '<PAD>'
UNK = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1
hidden_size = 768
device = "cuda" if torch.cuda.is_available() else "cpu"
label2idx_path = "./bertlabel_data/label2idx.json"

def load_json(data_path):
    with open(data_path,encoding="utf-8") as f:
        return json.loads(f.read())

def pred(input_ids, token_type_ids, attention_mask,label_lenth):
    logits,_ = model(input_ids, token_type_ids, attention_mask)
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
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--model_name', type=str, default="BertLabel")
    parser.add_argument('--encode_dim', type=int, default=768, help="bert encode_dim")
    parser.add_argument('--pad_size', type=int, default=512, help="max lenth")
    parser.add_argument('--num_filters', type=int, default=256, help="卷积核数量(channels数)")
    parser.add_argument('--a', type=float, default=1, help="merge a")
    parser.add_argument('--b', type=float, default=1, help="merge b")
    args = parser.parse_args()
    label2idx = load_json(label2idx_path)
    class_num = len(label2idx)
    model = Bert_label(hidden_size=hidden_size, class_num=class_num, args=args).to(device)
    model.load_state_dict(torch.load('./model/BERT_Label.pkl', map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained("./bert_model")  # 本地vocab
    max_len = 512
    label2idx = load_json("./bertlabel_data/label2idx.json")
    idx2label = {}
    for key, value in label2idx.items():
        idx2label[value] = key
    label_lenth = load_json("./bertlabel_data/label_len")
    inp = input("请输入裁判文书事实描述：")
    fact = inp.replace("\\n", "").replace(" ", "")
    tokenizers = tokenizer(fact, padding=True, truncation=True, max_length=max_len, return_tensors="pt",
                           is_split_into_words=False)
    input_ids = tokenizers["input_ids"].to(device)
    token_type_ids = tokenizers["token_type_ids"].to(device)
    attention_mask = tokenizers["attention_mask"].to(device)
    out = pred(input_ids, token_type_ids, attention_mask, label_lenth)
    print(out)