import math


def get_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    return lines


def get_result(lines):
    results = []
    for line in lines:
        results.append(line.split())

    return results


def arg_result(datas):
    r = []
    for data in datas:
        r.append(float(data))

    max_ = max(r)
    index = r.index(max_)

    if index == 0:
        return ('Neutral:', max_)

    elif index == 1:
        return ('Positive:', max_)

    else:
        return ('Negative:', max_)


def full_result(datas):
    r = []
    dict = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    for i, data in enumerate(datas):
        r.append((dict[i], float(data)))

    return r

def output(texts,full=False):
    r=[]
    for text in texts:
        if full:
            r.append(full_result(text))
        else:
            r.append(arg_result(text))

    return r

# def test():
#     lines = get_data('../predict_bert_function/output/test_results.tsv')
#     r = get_result(lines)
#     print(output(r))
#
#
# if __name__ == '__main__':
#     test()
