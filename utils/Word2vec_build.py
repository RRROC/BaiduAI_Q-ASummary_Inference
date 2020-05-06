import multiprocessing
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import pickle
import os


def dump_pkl(vocab, pkl_path, overwrite=True):
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('save %s ok' % pkl_path)


def load_pkl(pkl_save_path):
    load_file = open(pkl_save_path, "rb")
    word_dic = pickle.load(load_file)
    return word_dic


def __read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_path, train_y_path, test_x_path):
    result = []
    lines = __read_lines(train_x_path)
    lines += __read_lines(train_y_path)
    lines += __read_lines(test_x_path)

    for line in lines:
        result.append(line)
    return result


def save_sentence(lines, sentence_path):
    with open(sentence_path, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s' % line.strip())

    print('save sentence in %s' % sentence_path)


# Wk2 homework
# 使用gensim word2vec来训练词向量
def build_by_word2vec(train_x_path, train_y_path, test_x_path, sentence_out_path='',
                      w2v_bin_path='../resource/model/w2v.bin', min_count=50):
    sentences = extract_sentence(train_x_path, train_y_path, test_x_path)
    save_sentence(sentences, sentence_out_path)
    print('train w2v model...')

    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_out_path),
                   size=256, window=5, min_count=min_count, iter=5, workers=multiprocessing.cpu_count())

    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print('w2v model has been finished!')


# Wk2 homework
# 使用gensim fasttext来训练词向量
def build_by_fasttext(train_x_path, train_y_path, test_x_path, sentence_out_path='',
                      ft_bin_path='../resource/model/ft.bin', min_count=50):
    sentences = extract_sentence(train_x_path, train_y_path, test_x_path)
    save_sentence(sentences, sentence_out_path)
    print('train fasttext model...')
    ft = FastText(sg=1, sentences=LineSentence(sentence_out_path), size=256, window=5, min_count=min_count,
                  workers=multiprocessing.cpu_count(), iter=5)

    ft.wv.save_word2vec_format(ft_bin_path, binary=True)
    print('fasttext model has been finished!')


# 读取词向量和Wk1构建的vocab词表，以vocab中的index为key值构建embedding_matrix
def create_word2vec_metric(vocab_path, model_path):
    vocab = []
    with open(vocab_path, mode='r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.split())

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    word2vec_metric = {}
    for word in model.vocab:
        for ele in vocab:
            if word == ele[1]:
                word2vec_metric[int(ele[0])] = model[word]
                break

    return word2vec_metric


if __name__ == '__main__':
    # 通过word2vec训练词向量
    build_by_word2vec('../resource/output/train_set_x.txt', '../resource/output/train_set_y.txt',
                      '../resource/output/test_set_x.txt', sentence_out_path='../resource/output/sentences.txt')
    # 通过fasttext训练词向量
    build_by_fasttext('../resource/output/train_set_x.txt', '../resource/output/train_set_y.txt',
                      '../resource/output/test_set_x.txt', sentence_out_path='../resource/output/sentences.txt')

    # Wk2 homework
    # 生成词向量字典
    w2v_metric = create_word2vec_metric('../resource/output/vocab.txt', '../resource/model/w2v.bin')
    ft_metric = create_word2vec_metric('../resource/output/vocab.txt', '../resource/model/ft.bin')

    print('length of w2v_metrics: {}'.format(len(w2v_metric)))
    print('length of ft_metrics: {}'.format(len(ft_metric)))

    # Wk2 homework
    # 存储词向量字典
    dump_pkl(w2v_metric, '../resource/output/w2v_vocab_metric.txt')
    dump_pkl(ft_metric, '../resource/output/ft_vocab_metric.txt')

    # 测试两种model的表现
    w2v_model = KeyedVectors.load_word2vec_format('../resource/model/w2v.bin', binary=True)
    ft_model = KeyedVectors.load_word2vec_format('../resource/model/ft.bin', binary=True)
    print('w2v_model\'s most similar word: \n{}'.format(w2v_model.most_similar('汽车')))
    print('ft_model\'s most similar word: \n{}'.format(ft_model.most_similar('汽车')))

