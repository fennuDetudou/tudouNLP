import os

class tools(object):
    def __init__(self):
        self.work_path=os.getcwd()
        self.module_path=os.path.dirname(__file__)

        self.path=os.path.abspath(self.work_path)

    def posseg_data(self,input_dir,output_file,combine=False,
                    single_line=False,log=False,max_len=128):
        '''
        将使用词性标注的文件转换为用BIS分块标记的文件。
        :param input_dir: 指定存放语料库的文件夹，程序将会递归查找目录下的文件。
        :param output_dir: 指定标记好的文件的输出路径。
        :param combine: 是否组装为一个文件
        :param single_line: 是否为单行模式
        :param log: 是否打印进度条
        :param max_len: 处理后的最大语句长度（将原句子按标点符号断句，若断句后的长度仍比最大长度长，将忽略
        :return:
        '''
        input_dir1=os.path.abspath(input_dir)
        output=os.path.abspath(output_file)
        try:
            os.chdir(self.module_path)
            os.system("python convert_to_tagdata.py {} {}".format(input_dir1,output))
        finally:
            os.chdir(self.work_path)

    def compress_model(self,input_file,output_file):
        '''
        参数为文件名称而不是目录名称
        :param input_file:
        :param output_file:
        :return:
        '''
        input1=os.path.abspath(input_file)
        output=os.path.abspath(output_file)
        print(output)

        try:
            os.chdir(self.module_path)
            os.system("python compress_ckpt.py --input_file={} --output_file={}".format(input1,output))
        finally:
            os.chdir(self.work_path)



