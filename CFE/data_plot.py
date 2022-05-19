import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def data_label_count(data_path):
    f = open(data_path,encoding='utf-8')
    t = json.load(f)
    JKJFXS = {}
    JKRJBSX = {}
    JKYT ={}
    JKHYDPJ ={}
    CJRJBSX = {}
    CJYT = {}
    DBLX = {}
    YDQNLL = {}
    YDJXFS = {}
    HKXS = {}
    max_len = 0
    for i,v in enumerate(t):
        if len(v['fact'])>max_len:
            max_len = len(v['fact'])
        value = list(v["attr"].values())
        for i in range(len(value[0])):
            num1 = JKJFXS.get(value[0][i], 0) + 1
            JKJFXS[value[0][i]] = num1
        for i in range(len(value[1])):
            num2 = JKRJBSX.get(value[1][i], 0) + 1
            JKRJBSX[value[1][i]] = num2
        for i in range(len(value[2])):
            num3 = JKYT.get(value[2][i], 0) + 1
            JKYT[value[2][i]] = num3
        for i in range(len(value[3])):
            num4 = JKHYDPJ.get(value[3][i], 0) + 1
            JKHYDPJ[value[3][i]] = num4
        for i in range(len(value[4])):
            num5 = CJRJBSX.get(value[4][i], 0) + 1
            CJRJBSX[value[4][i]] = num5
        for i in range(len(value[5])):
            num6= CJYT.get(value[5][i], 0) + 1
            CJYT[value[5][i]] = num6
        for i in range(len(value[6])):
            num7 = DBLX.get(value[6][i], 0) + 1
            DBLX[value[6][i]] = num7
        for i in range(len(value[7])):
            num8 = YDQNLL.get(value[7][i], 0) + 1
            YDQNLL[value[7][i]] = num8
        for i in range(len(value[8])):
            num9 = YDJXFS.get(value[8][i], 0) + 1
            YDJXFS[value[8][i]] = num9
        for i in range(len(value[9])):
            num10 = HKXS.get(value[9][i], 0) + 1
            HKXS[value[9][i]] = num10
    return JKJFXS,JKRJBSX,JKYT,JKHYDPJ,CJRJBSX,CJYT,DBLX,YDQNLL,YDJXFS,HKXS

def label_plot(dict):
    matplotlib.rcParams['font.family'] = 'SimHei'
    y = dict.values()
    N = len(dict)
    x = np.arange(N)
    plt.xlabel('要素标签')
    plt.ylabel('数量')
    # str1 = str(dict.keys())
    plt.bar(x, height=y, width=0.5,label="担保类型")
    # 添加数据标签
    for a, b in zip(x, y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=14)
    # 添加图例
    plt.legend()
    plt.show()

def plot_(dict1,dict2):
    matplotlib.rcParams['font.family'] = 'SimSun'
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    # x1 = list(dict.keys(dict1))
    # x2 = list(dict.keys(dict2))

    x1 = ['现金', '未知或模糊', '银行转账', '其他', '网络平台', '网上汇款', '未出借', '票据', '授权账户']
    x2 = ['合同、借条', '担保', '欠条', '收据、收条', '还款承诺', '其他', '聊天记录', '未知或模糊']

    y1 = dict.values(dict1)
    y2 = dict.values(dict2)

    ax0.bar(x1, height=y1, width=0.5, color='dodgerblue')
    ax0.set_xticklabels(x1, fontsize=14,rotation=35)
    ax0.set_yticklabels([0,250,500,750,1000,1250,1500,1750], fontsize=14)
    ax0.set_title('借款交付形式', fontsize=16, loc='center')
    for a, b in zip(x1, y1):
        ax0.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=13)

    ax1.bar(x2, height=y2, width=0.5, color='dodgerblue')
    ax1.set_xticklabels(x2, fontsize=14, rotation=35)
    ax1.set_yticklabels([0,500,1000,1500,2000,2500,3000], fontsize=14)
    ax1.set_title('借贷合意的凭据',fontsize = 16, loc='center')
    for a, b in zip(x2, y2):
        ax1.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=13)
    plt.show()


if __name__ == '__main__':
    JKJFXS,JKRJBSX,JKYT,JKHYDPJ,CJRJBSX,CJYT,DBLX,YDQNLL,YDJXFS,HKXS = data_label_count("./raw_data.json")
    # label_plot(DBLX) # 绘制单独属性的标签分布
    plot_(JKJFXS,JKHYDPJ)   #绘制部分属性的统计分布
