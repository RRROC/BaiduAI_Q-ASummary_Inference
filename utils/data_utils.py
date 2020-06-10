import numpy as np
import pickle
import os


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def load_word2vec(params, max_vocab_size=30000):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return:
    """
    word2vec_dict = load_pkl(params['word2vec_output'])
    vocab_dict = open(params['vocab_path'], encoding='utf-8').readlines()
    embedding_matrix = np.zeros((max_vocab_size, params['embed_size']))

    for i in range(max_vocab_size):
        embedding_matrix[i] = word2vec_dict[i]
    # # 加入MASK的embedding
    # for i in range(4):
    #     embedding_matrix[i] = word2vec_dict[i]
    #
    # for line in vocab_dict[:params['vocab_size'] - 4]:
    #     word_id = line.split()
    #     i, word = word_id
    #     embedding_vector = word2vec_dict.get(int(i) + 4)
    #     if embedding_vector is not None:
    #         embedding_matrix[int(i) + 4] = embedding_vector
    #
    # unk_matrix = embedding_matrix[1]
    #
    # for i in range(params['vocab_size'], max_vocab_size):
    #     embedding_matrix[i] = unk_matrix

    return embedding_matrix


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)
