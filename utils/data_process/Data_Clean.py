import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba

REMOVE_WORDS = ['|', '[', ']', '语音', '图片']


# 建立停止词的集合set
def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


# 把原始数据中的不需要的词删除掉
def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


# 通过jieba分割语句
def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for word in word_seq:
                w_p = posseg.lcut(word)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


# 读取数据集，并分词
def parse_data(train_path, test_path):
    # 读取csv
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    # 去掉report列为空的行
    train_df.dropna(subset=['Report'], how='any', inplace=True)

    # 剩余字段是输入，包含Brand,Model,Question,Dialogue，如果有空，填充即可
    train_df.fillna('', inplace=True)

    # 实际输入只选择Question & Dialogue两个列
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = train_df.Report
    assert len(train_x) == len(train_y)

    # 处理测试数据集
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []

    return train_x, train_y, test_x, test_y


def save_data(train_x, train_y, test_x, train_x_output, train_y_output, test_x_output, stop_words_path):
    stop_words = read_stopwords(stop_words_path)
    print('start to cut the words...')
    with open(train_x_output, mode='w', encoding='utf-8') as f_train_x:
        count_train_x = 0
        for line in train_x:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                # 考虑停词
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f_train_x.write('%s' % seg_line)
                    f_train_x.write('\n')
                    count_train_x += 1
        print('train_x_length: ', count_train_x)

    with open(train_y_output, 'w', encoding='utf-8') as f_train_y:
        count_2 = 0
        for line in train_y:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                # seg_list = remove_words(seg_list)
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f_train_y.write('%s' % seg_line)
                    f_train_y.write('\n')
                else:
                    f_train_y.write("随时 联系")
                    f_train_y.write('\n')
                    # print('11111')
                count_2 += 1
        print('train_y_length is ', count_2)

    with open(test_x_output, 'w', encoding='utf-8') as f_test_x:
        count_3 = 0
        for line in test_x:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                seg_list = [word for word in seg_list if word not in stop_words]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f_test_x.write('%s' % seg_line)
                    f_test_x.write('\n')
                    count_3 += 1
        print('test_y_length is ', count_3)


if __name__ == '__main__':
    train_x, train_y, test_x, _ = parse_data('../../resource/input/AutoMaster_TrainSet.csv',
                                             '../resource/input/AutoMaster_TestSet.csv')

    print(len(train_x))
    print(len(train_y))

    save_data(train_x, train_y, test_x, '../resource/output/train_set_x.txt', '../resource/output/train_set_y.txt',
              '../resource/output/test_set_x.txt', stop_words_path='../../resource/input/stop_words.txt')
