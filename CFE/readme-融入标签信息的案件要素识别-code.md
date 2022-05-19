

# 融入标签信息的案件要素识别-code

硬件环境必备：GPU （参考：**Tesla k80 **, **cuda-v:**11.2)

## 终端操作：

### 1、配置环境：(py3.6)

```shell
pip install -r requirements.txt
```

### 2、数据获取、预处理

- 获取数据、数据集划分、查看裁判文书长度：

  ```shell
  python data_get.py --process_obj get_data  #划分数据集
  python data_get.py --process_obj len_conut  #查看数据集长度
  ```

- 数据集标签信息绘图：

  ```shell
  python data_plot.py
  ```

- 数据预处理：

  ```shell
  python data_preprocess.py --use_word --data_path data  #为基于词输入的模型准备数据
  python data_preprocess.py --data_path bert_data #为基于字符输入的模型准备数据
  python data_preprocess.py --data_path bertlabel_data #为本文模型准备数据
  ```

### 3、模型训练与测试

```shell
python train.py --model_name TextCNN #训练baseline（Fastext、DPCNN、bert、bert-cnn、bert-lstm、bert-ms）调优详见 5、
python train_label.py --model_name BertLabel #训练、验证、测试模型（参数配置详见parser）
```

### 4、预测

```shell
python pred.py
*******************使用展示**********************
完整裁判文书为: "原告：刘少选，男，1970年9月13日出生，汉族，住洛阳市汝阳县。 被告：李庆杰，又名李国军，男，1963年2月4日出生，汉族，住济源市。 委托代理人：王有龙，系社区推荐。\n\n 原告刘少选向本院提出诉讼请求：依法判令被告支付其11.8万元。事实和理由：2016年4月6日，被告于其签订邵原机械厂六层砖混楼工程承包合同，被告要求其先交合同押金50000元，承诺三天内开始施工，后来却以各种借口与理由不开工；2016年7月28日，在其与被告多次沟通后，被告却说需要先结清工地施工以前的费用，然后才能开工，并要求其再次给被告68000元用于周转，同时承诺一个月内将这68000元连本带息还给其，可之后，被告不再接听电话，也未还款，更没有让其施工。 被告李庆杰辩称，原告所述属实，对数额无异议。 当事人围绕诉讼请求依法提交了证据，双方对如下证据的真实性无异议，本院予以确认并在卷佐证：1、2016年4月6日，被告出具的收到条一份；2、2017年1月28日被告出具的借据一份；3、中国农业银行回单一份；4、建筑工程施工项目承包合同一份。 本院经审理认定事实如下：被告李庆杰为济源市邵原机械制造厂的法定代表人。2016年4月6日，原告刘少选及他人姚少伟与济源市邵原机械制造厂签订建筑工程施工项目承包合同。原告刘少选及他人姚少伟按被告要求交合同押金50000元，后一直至今未开工。2016年7月28日，被告向原告借款68000元，并给原告出具借据一份，载明：“今借到刘少选现金陆万捌仟元整68000.00元。》李国军2017.7.28号”。该款至今未还。 \n"
！！！在使用时需要去除涉诉人情况，如下所示！！！
 》》'python pred.py'
 》》'请输入裁判文书事实描述：'原告刘少选向本院提出诉讼请求：依法判令被告支付其11.8万元。事实和理由：2016年4月6日，被告于其签订邵原机械厂六层砖混楼工程承包合同，被告要求其先交合同押金50000元，承诺三天内开始施工，后来却以各种借口与理由不开工；2016年7月28日，在其与被告多次沟通后，被告却说需要先结清工地施工以前的费用，然后才能开工，并要求其再次给被告68000元用于周转，同时承诺一个月内将这68000元连本带息还给其，可之后，被告不再接听电话，也未还款，更没有让其施工。 被告李庆杰辩称，原告所述属实，对数额无异议。 当事人围绕诉讼请求依法提交了证据，双方对如下证据的真实性无异议，本院予以确认并在卷佐证：1、2016年4月6日，被告出具的收到条一份；2、2017年1月28日被告出具的借据一份；3、中国农业银行回单一份；4、建筑工程施工项目承包合同一份。 本院经审理认定事实如下：被告李庆杰为济源市邵原机械制造厂的法定代表人。2016年4月6日，原告刘少选及他人姚少伟与济源市邵原机械制造厂签订建筑工程施工项目承包合同。原告刘少选及他人姚少伟按被告要求交合同押金50000元，后一直至今未开工。2016年7月28日，被告向原告借款68000元，并给原告出具借据一份，载明：“今借到刘少选现金陆万捌仟元整68000.00元。》李国军2017.7.28号”。该款至今未还。
 》》'Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\Lenovo\AppData\Local\Temp\jieba.cache
Loading model cost 0.703 seconds.
Prefix dict has been built successfully.''
》》{'借款交付形式': '银行转账', '借款人基本属性': '借款法人', '借款用途': '企业生产经营', '借贷合意的凭据': '担保', '出借人基本属性': '还款法人', '出借意图': '其他出借意图', '担保类型': '抵押', '约定期内利率': '36%（不含）以上', '约定计息方式': '其他计息方式', '还款交付形式': '票据还款'}
》》'Process finished with exit code 0'
```

### 5、调优

TextCNN：

```python
learning_rate = 1e-3 
epoch = 20
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)     # 设置学习率下降策略
```

DPCNN：

```python
learning_rate = 1e-3 
epoch = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.001)     # 设置学习率下降策略
```

BiLSTM-Att（TextRNN_Att）：

```python
learning_rate = 1e-2 
epoch = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

BERT-CNN：

```python
learning_rate = 4e-5
dropout = 0.1
epoch = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)     # 设置学习率下降策略
```

fastext：

```python
learning_rate = 1e-2 
epoch = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

bert-ms：

```python
learning_rate = 4e-5 
epoch = 20
bsz = 12
dropout = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

Roberta：

```python
learning_rate = 4e-5 
dropout = 0.5
bsz = 12
epoch = 20
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

BERT-Base:

```python
learning_rate = 4e-5 
dropout = 0.5
bsz = 12
epoch = 20
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

BERT-LSTM

```python
learning_rate = 4e-5 
dropout = 0.5
bsz = 12
epoch = 20
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)     # 设置学习率下降策略
```

