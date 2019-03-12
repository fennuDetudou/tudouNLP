def get_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines


def proecess_data(texts):
    line = []
    setence = []
    lines=[]
    for word in texts:
        if word.strip() == '[CLS]':
            continue
        if word.strip() == '[SEP]':
            line.append(setence)
            setence = []
            continue
        setence.append(word.strip())

    return line


def segmentor(texts, labels):

    cut_result=[]
    posseg_result=[]


    for text,label in zip(texts,labels):
        text_ = []
        label_ = []

        t, l = [], []
        i = 0
        while (i < len(text)):
            if label[i].startswith(('B', 'S')):
                text_.append(t)
                label_.append(l)
                t = []
                l = []
                t.extend(text[i])
                l.extend(label[i].split('-')[-1])
            elif label[i].startswith('I'):
                t.extend(text[i])
            i += 1
        text_.append(t)
        label_.append(l)

        t_result = text_[1:]
        l_result = label_[1:]

        sentence=[]
        posseg=[]
        for i, word in enumerate(t_result):
            sentence.append(''.join(word))
            posseg.append(''.join(l_result[i]))

        cut_result.append(sentence)
        posseg_result.append(posseg)

    results=[]
    for c,p in zip(cut_result,posseg_result):
        results.append(list(zip(c,p)))

    return cut_result,results


# def test():
#     text_file = '../predict_bert_function/output/token_test.txt'
#     label_file = '../predict_bert_function/output/label_test.txt'
#     texts = get_data(text_file)
#     label = get_data(label_file)
#     texts = proecess_data(texts)
#     labels = proecess_data(label)
#
#     a = segmentor(texts, labels)
#     print(a)
#
#
# if __name__ == '__main__':
#     test()
