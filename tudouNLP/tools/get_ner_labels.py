def get_labels(file):
    labels=set()
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            try:
                labels.add(line.split()[1])
            except:
                pass

    return list(labels)

if __name__ == '__main__':
    labels=get_labels('../../datasets/train.txt')
    print(labels)