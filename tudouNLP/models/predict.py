import os
import shutil

from tools import ner_predict_utils
from tools import tag_predict_utils
from tools import sentiment_predict_utils


class tagger(object):

    def __init__(self, model_dir):
        self._work_path = os.getcwd()
        self._module_path = os.path.dirname(__file__)
        self._model_dir = os.path.abspath(model_dir)

    def _data_process(self, document):
        try:
            os.chdir(self._module_path)
            if not os.path.exists('./output'):
                os.mkdir('./output')
            with open('./output/test.txt', 'w+', encoding='utf-8') as f:
                for sentence in document:
                    for word in sentence:
                        f.write(word + '\n')
                    f.write('\n')
        finally:
            os.chdir(self._work_path)

    def ner(self, document):
        model_path = os.path.join(self._model_dir, 'ner_model/model')
        if type(document) is str:
            raise Exception('the document must be list or tuple')
        self._data_process(document)
        try:
            os.chdir(self._module_path)
            os.system('python pos_tag.py \
                        --task_name=ner \
                        --do_predict=true \
                        --data_dir=./output \
                        --vocab_file=./vocab.txt \
                        --bert_config_file=./bert_config.json \
                        --init_checkpoint={} \
                        --max_seq_length=128 \
                        --train_batch_size=32 \
                        --learning_rate=2e-5 \
                        --output_dir=./output'.format(model_path))

            text_file = './output/token_test.txt'
            label_file = './output/label_test.txt'

            texts = ner_predict_utils.get_data(text_file)
            labels = ner_predict_utils.get_data(label_file)
            texts = ner_predict_utils.proecess_data(texts)
            labels = ner_predict_utils.proecess_data(labels)

            result = ner_predict_utils.recognized_entity(texts, labels)

            shutil.rmtree('./output')

        finally:
            os.chdir(self._work_path)

        return result

    def cut(self, document, mode='cut'):
        model_path = os.path.join(self._model_dir, 'tag_model/model')
        if type(document) is str:
            raise Exception('the document must be list or tuple')
        self._data_process(document)

        try:
            os.chdir(self._module_path)

            os.system('python pos_tag.py \
                               --task_name=tag \
                               --do_predict=true \
                               --data_dir=./output \
                               --vocab_file=./vocab.txt \
                               --bert_config_file=./bert_config.json \
                               --init_checkpoint={} \
                               --max_seq_length=128 \
                               --train_batch_size=32 \
                               --learning_rate=2e-5 \
                               --output_dir=./output'.format(model_path))

            text_file = './output/token_test.txt'
            label_file = './output/label_test.txt'
            texts = tag_predict_utils.get_data(text_file)
            label = tag_predict_utils.get_data(label_file)
            texts = tag_predict_utils.proecess_data(texts)
            labels = tag_predict_utils.proecess_data(label)

            cut, posseg = tag_predict_utils.segmentor(texts, labels)
            shutil.rmtree('./output')

        finally:
            os.chdir(self._work_path)

        if mode == 'cut':
            return cut
        elif mode == 'posseg':
            return posseg
        else:
            raise Exception("the mode must be cut or pooseg !")


class sentence(object):

    def __init__(self, model_dir):

        self._work_path = os.getcwd()
        self._module_path = os.path.dirname(__file__)
        self._model_dir = os.path.abspath(model_dir)

    def _data_process(self, document):

        try:
            os.chdir(self._module_path)
            if not os.path.exists('./output'):
                os.mkdir('./output')

            with open('./output/test.txt', 'w+', encoding='utf-8') as f:
                for sentence in document:
                    f.write(sentence)
                    f.write('\n')
        finally:
            os.chdir(self._work_path)

    def sentiment(self, document, full_msg=False):
        model_path = os.path.join(self._model_dir, 'sentiment_model/model')
        if type(document) is str:
            raise Exception('the document must be list or tuple')

        self._data_process(document)
        try:
            os.chdir(self._module_path)

            os.system('python sentence.py \
                          --task_name=classify \
                          --do_predict=true \
                          --data_dir=./output \
                          --vocab_file=./vocab.txt \
                          --bert_config_file=./bert_config.json \
                          --init_checkpoint={} \
                          --max_seq_length=128 \
                          --train_batch_size=32 \
                          --learning_rate=2e-5 \
                          --output_dir=./output'.format(model_path))

            lines = sentiment_predict_utils.get_data('./output/test_results.tsv')
            lines = sentiment_predict_utils.get_result(lines)
            results = sentiment_predict_utils.output(lines, full_msg)

            shutil.rmtree('./output')
            return results
        except Exception as e:
            print(e)
        finally:
            os.chdir(self._work_path)

    def pair(self, document, full_msg=False, model_name=None):
        if type(document) is str:
            raise Exception('the document must be list or tuple')
        model_path = os.path.join(self._model_dir, model_name)
        self._data_process(document)
        try:
            os.chdir(self._module_path)

            os.system('python sentence.py \
                          --task_name=pair \
                          --do_predict=true \
                          --data_dir=./output \
                          --vocab_file=./vocab.txt \
                          --bert_config_file=./bert_config.json \
                          --init_checkpoint={} \
                          --max_seq_length=128 \
                          --train_batch_size=32 \
                          --learning_rate=2e-5 \
                          --output_dir=./output'.format(model_path))

            lines = sentiment_predict_utils.get_data('./output/test_results.tsv')
            lines = sentiment_predict_utils.get_result(lines)
            results = sentiment_predict_utils.output(lines, full_msg)

            shutil.rmtree('./output')
            return results
        except Exception as e:
            print(e)
        finally:
            os.chdir(self._work_path)



