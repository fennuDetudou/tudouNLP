import tqdm
import argparse


def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        text=f.readlines()

    return text


def convert(input_file,output_file):
    texts=read_file(file=input_file)
    with open(output_file,'w+',encoding='utf-8') as f:
        for i in tqdm.tqdm(range(len(texts))):
            text, tagged = texts[i].split('\t')
            words = text.strip().split(' ')
            tags = tagged.strip().split(' ')

            for word, tag in zip(words, tags):
                f.write(word+' '+tag+'\n')
            f.write('\n')
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="将分词标注数据转换为bert可以读取的数据格式")
    parser.add_argument('input_file',type=str,help="输入文件")
    parser.add_argument('output_file',type=str,help="输出文件")

    args=parser.parse_args()
    convert(args.input_file,args.output_file)
