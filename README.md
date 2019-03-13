# 说明

1. 基于bert的中文自然语言处理工具
2. 包括情感分析、中文分词、词性标注、以及命名实体识别功能
3. 提供了训练接口，通过指定输入输出以及谷歌提供的下载好的预训练模型即可进行自己的模型的训练，训练任务有task_name参数决定，目前提供的任务主要包括句子匹配、文本分类、命名实体识别、序列标注任务
4. 使用`pip install tudou`安装使用
5. 需要下载预先训练好的模型，模型地址在底部

# 使用示例
在predict_test.ipynb中有预测代码的演示，可以看到效果比绝大多数开源中文nlp库要好，但是速度较慢（时间主要浪费在加载模型参数上了），**所以推荐一次性输入多个语句的文本列表进行使用**

# 依赖项
## 依赖项

`python >=3.6
tensorflow >= 1.12.0`
## 硬件
1. 预测与使用在普通cpu机器上既可以运行
2. 训练接口需要在GPU机器上进行，当内存不够用时，推荐减少batch_size而不是max_sequence_len,对精度影响较小

# 功能简介

提供了三个接口，包括预测，常用工具以及利用bert训练模型的接口

# 使用说明

1. 在使用前要先下载模型，模型下载地址附在最后，根据不同的任务下载不同的模型
1. 同时也可以使用自己训练好的模型
1. 训练模型需要预先下载谷歌提供的bert预训练模型（该项目也提供：位于pre_trained_model下

## train函数

新建一个实例

`trainer=tudouNLP.models.train.train(*params)`

### help

train函数的说明函数，包括一些参数及文件格式的说明

`trainer.help()`

### 训练

`trainer()`

### 预测

`results=trainer.predict()`

#### 说明

1. 训练的时候没有返回值，根据参数中task_name开始不同的训练任务

1. 包括文本分类，序列标注以及句子匹配任务的训练

1. 预测时要注意与训练时参数要相同（主要是`label_list、label_dict`），同时输出目录也要相同

1. 参数简介

   ```
    :param task_name:任务名：目前包括实体识别ner，序列标注tag，句子分类classify，句子配对pair
    :param label_list: 任务的标签列表，在序列标注任务中要加入【CLS】,[SEP]
    :param label_dict: 序列标注任务中标签与ID对应的字典名
    :param data_dir: 数据文件
    :param model_dir: 模型文件
    :param output_dir: 输出文件
    :param eval: 是否进行验证
    :param max_seq_length:
    :param learning_rate:
    :param batch_size:
   ```

1. 提供的文件格式说明

   ```
   1. 序列标注任务文件格式为 word tag
   2. 文本分类任务文件格式为 sentence label
   3. 句子配对任务文件格式为 index text1 text2 label  ，其中index为不必要的列，中间分隔符为\t
   4. 文件在data_dir中，训练文件命名为train.txt，验证集文件命名为dev.txt
   ```

## predict

**使用时要创建一个实例**


### sentence函数

`predictor=tudouNLP.models.predict.sentence(model_dir）# 参数为模型所在文件夹`

#### 情感分析

`result=predictor.sentiment(document,full_msg)`

1. 返回情感分析结果，当full_msg参数为True时，返回全部的分析结果
1. document为要分析的句子列表
1. **注意：即使是单个句子，也要以列表或元组的形式输入**

#### 句子匹配

`result=predictor.pair(document,full_msg,model_name)`

同情感分析

### tagger函数

`predictor=tudouNLP.models.predict.tagger(model_dir)`

#### 分词

`result=predictor.cut(document,mode='cut')`

1. 返回分词结果列表
1. document为要分析的句子列表
1. 注意：即使是单个句子，也要以列表或元组的形式输入

#### 词性标注

`result=predictor.cut(document,mode='posseg')`

同分词，不过返回分词结果列表

## utils函数

**使用时要创建一个实例**

`tool=tudouNLP.tools.utils.tools()`

#### 序列标注数据集转换

`tool.posseg_data(input_dir,output_file)`

1. 将序列标注的数据集转换为bert模型可以识别的模式

#### 模型压缩

`tool.compress_model(input_file,output_file)`

将训练后的模型参数进行压缩

# 模型下载

链接: https://pan.baidu.com/s/1_dBX3-mjY3-Dedm96XNY2g     提取码: tjqe 







