from gensim.models import Word2Vec
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


def build(train_x_path, train_y_path, test_x_path, out_path=None, sentence_out_path='',
          w2v_bin_path='./model/w2v.bin', min_count=200):
    sentences = extract_sentence(train_x_path, train_y_path, test_x_path)
    save_sentence(sentences, sentence_out_path)
    print('train w2v model...')

    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_out_path),
                   size=256, window=5, min_count=min_count, iter=5)

    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)

    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)

    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]

    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('./resource/output/train_set_x.txt', './resource/output/train_set_y.txt', './resource/output/test_set_x.txt',
          out_path='./resource/output/word2vec.txt', sentence_out_path='./resource/output/sentences.txt')
