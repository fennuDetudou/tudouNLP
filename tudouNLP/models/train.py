import os

class train(object):
    def __init__(self,task_name,data_dir,model_dir,output_dir,
                 label_dict='label2id.pkl',label_list=None,
                 eval=True,max_seq_length=128,
                 learning_rate=2e-5,batch_size=32):
        '''
        训练函数
        :param task_name:任务名：目前包括实体识别ner，序列标注tag，句子分类classify，句子配对pair
        :param label_list:任务的标签序列
            1. ner，tag,classify 都需要这个参数
            2. 在序列标注任务中要加入【CLS】,[SEP]
        :param label_dict: 序列标注任务中标签与ID对应的字典名
        :param data_dir: 数据文件
        :param model_dir: 模型文件
        :param output_dir: 输出文件
        :param eval: 是否进行验证
        :param max_seq_length:
        :param learning_rate:
        :param batch_size:
        '''
        self.task_name=task_name
        if label_list:
            self.label_list=','.join(label_list)
        else:
            self.label_list=None
        self.label_dict=label_dict
        self.data_dir=os.path.abspath(data_dir)
        model_dir1=os.path.abspath(model_dir)
        self.model_dir=os.path.join(model_dir1,'bert_model.ckpt')

        work_path=os.getcwd()

        self.output_dir=os.path.abspath(output_dir)
        self.eval=eval
        self.max_seq_length=max_seq_length
        self.lr=learning_rate
        self.batch_size=batch_size

        module_path = os.path.dirname(__file__)
        os.chdir(module_path)
        try:
            self._train()
        finally:
            os.chdir(work_path)

    def _train(self):

       if self.eval:
           os.system('python run_classifier.py \
                      --label_dict={} \
                      --label_list={} \
                      --task_name={} \
                      --do_train=true \
                      --do_eval=true \
                      --data_dir={} \
                      --vocab_file=./vocab.txt \
                      --bert_config_file=./bert_config.json \
                      --init_checkpoint={} \
                      --max_seq_length={} \
                      --train_batch_size={} \
                      --learning_rate={} \
                      --output_dir={}'.format(self.label_dict,
                                              self.label_list,
                                              self.task_name,
                                              self.data_dir,
                                              self.model_dir,
                                              self.max_seq_length,
                                              self.batch_size,
                                              self.lr,
                                              self.output_dir))
       else:
           os.system('python run_classifier.py \
                                 --label_dict={} \
                                 --label_list={} \
                                 --task_name={} \
                                 --do_train=true \
                                 --data_dir={} \
                                 --vocab_file=./vocab.txt \
                                 --bert_config_file=./bert_config.json \
                                 --init_checkpoint={} \
                                 --max_seq_length={} \
                                 --train_batch_size={} \
                                 --learning_rate={} \
                                 --output_dir={}'.format(self.label_dict,
                                                         self.label_list,
                                                         self.task_name,
                                                         self.data_dir,
                                                         self.model_dir,
                                                         self.max_seq_length,
                                                         self.batch_size,
                                                         self.lr,
                                                         self.output_dir))
    @ staticmethod
    def help():
        print('''训练函数——参数解析：
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
        ''')
        print('''文件格式说明：
        1. 序列标注任务文件格式为 word tag
        2. 文本分类任务文件格式为 sentence label
        3. 句子配对任务文件格式为 index text1 text2 label  ，其中index为不必要的列，中间分隔符为\t
        ''')



