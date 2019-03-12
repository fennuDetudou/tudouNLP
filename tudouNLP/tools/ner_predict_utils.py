def get_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines


def proecess_data(texts):
    lines = []
    setence = []
    for word in texts:
        if word.strip() == '[CLS]':
            continue
        if word.strip() == '[SEP]':
            lines.append(setence)
            setence = []
            continue
        setence.append(word.strip())

    return lines


def recognized_entity(texts, labels):

    result=[]

    for text,label in zip(texts,labels):
        p,l,o=[],[],[]
        p_,l_,o_=[],[],[]
        i=0
        while (i<len(text)):

            if label[i].startswith('O'):
                '''BO IO'''
                if i !=0:
                    if label[i-1].endswith('PER'):
                        p_.append(''.join(p))
                        p = []
                    if label[i-1].endswith('LOC'):
                        l_.append(''.join(l))
                        l = []
                    if label[i-1].endswith('ORG'):
                        o_.append(''.join(o))
                        o = []

            if label[i].startswith(('B')):
                if i==0:
                    '''B'''
                    if label[i].endswith('PER'):
                        p.extend(text[i])
                    if label[i].endswith('LOC'):
                        l.extend(text[i])
                    if label[i].endswith('ORG'):
                        o.extend(text[i])
                else:
                    if label[i-1].startswith('O'):
                        '''OB'''
                        if label[i].endswith('PER'):
                            p.extend(text[i])
                        if label[i].endswith('LOC'):
                            l.extend(text[i])
                        if label[i].endswith('ORG'):
                            o.extend(text[i])
                    else:
                        '''IB BB'''
                        if label[i-1].endswith('PER'):
                            p_.append(''.join(p))
                            p=[]
                            if label[i].endswith('PER'):
                                p.extend(text[i])
                            if label[i].endswith('LOC'):
                                l.extend(text[i])
                            if label[i].endswith('ORG'):
                                o.extend(text[i])
                        if label[i-1].endswith('LOC'):
                            l_.append(''.join(l))
                            l = []
                            if label[i].endswith('PER'):
                                p.extend(text[i])
                            if label[i].endswith('LOC'):
                                l.extend(text[i])
                            if label[i].endswith('ORG'):
                                o.extend(text[i])
                        if label[i-1].endswith('ORG'):
                            o_.append(''.join(o))
                            o = []
                            if label[i].endswith('PER'):
                                p.extend(text[i])
                            if label[i].endswith('LOC'):
                                l.extend(text[i])
                            if label[i].endswith('ORG'):
                                o.extend(text[i])
            if label[i].startswith('I'):
                '''OI BI II'''
                if label[i].endswith('PER'):
                    p.extend(text[i])
                if label[i].endswith('LOC'):
                    l.extend(text[i])
                if label[i].endswith('ORG'):
                    o.extend(text[i])
            i+=1

        result.append((p_,l_,o_))


    return result


# def test():
#     text_file = "../predict_bert_function/output/token_test.txt"
#     label_file = "../predict_bert_function/output/label_test.txt"
#     texts = get_data(text_file)
#     labels = get_data(label_file)
#     texts = proecess_data(texts)
#     labels = proecess_data(labels)
#
#     print(recognized_entity(texts, labels))
#
#
# if __name__ == '__main__':
#     test()
