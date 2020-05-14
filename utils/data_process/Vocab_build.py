from collections import defaultdict


def save_word_dict(vocab, save_path):
    with open(save_path, mode='w', encoding='utf-8') as f:
        for line in vocab:
            i, w = line
            f.write("%d\t%s\n" % (i, w))


def read_data(train_x_path, train_y_path, test_x_path):
    with open(train_x_path, 'r', encoding='utf-8') as f1, \
            open(train_y_path, 'r', encoding='utf-8') as f2, \
            open(test_x_path, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words += line.split(' ')

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    result = []
    if sort:
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1

        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)

        for i, item in enumerate(dic):
            word = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(word)
    else:
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(i, w) for i, w in enumerate(result)]
    reverse_vocab = [(w, i) for i, w in enumerate(result)]

    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data('../../resource/output/train_set_x.txt',
                      '../resource/output/train_set_y.txt',
                      '../resource/output/test_set_x.txt')
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '../../resource/output/vocab.txt')
