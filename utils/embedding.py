import utils.Data_Clean as dc
import utils.Vocab_build as vb
import utils.Word2vec_build as wb
from utils.config.read_config import ReadConfig


# 字典类
class Vocab:
    def __init__(self, vocab_file, vocab_max_size=None):
        self.PAD_TOKEN = '<PAD>'  # 如果输入的样本的长度不及阈值，那么剩余的位置补PAD
        self.UNKNOWN_TOKEN = '<UNK>'  # 如果总共的语料库有5万个词，但是我们只选取了3万个词做embedding。当遇到embedding之外的词是，就标注成UNK
        self.START_TOKEN = '<START>'  # 每句话的开头和结尾输入start 和stop， 如果输入样本的长度大于阈值，则截取到阈值，同样输入start和stop
        self.STOP_TOKEN = '<STOP>'

        self.MASK = ['<PAD>', '<UNK>', '<START>', '<STOP>']
        self.MASK_COUNT = len(self.MASK)
        self.pad_token_index = self.MASK.index(self.PAD_TOKEN)
        self.unknown_token_index = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_index = self.MASK.index(self.START_TOKEN)
        self.stop_token_index = self.MASK.index(self.STOP_TOKEN)
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, vocab_file, vocab_max_size=None):
        vocab = {mask: index for index, mask in enumerate(self.MASK)}
        reverse_vocab = {index: mask for index, mask in enumerate(self.MASK)}

        for line in open(vocab_file, mode='r', encoding='utf-8').readlines():
            index, word = line.strip().split('\t')  # 去除每行最后面的\n，然后根据\t分开
            index = int(index)
            if vocab_max_size and index > vocab_max_size - self.MASK_COUNT - 1:
                break
            vocab[word] = index + self.MASK_COUNT
            reverse_vocab[index + self.MASK_COUNT] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, id):
        if id not in self.id2word:
            return self.UNKNOWN_TOKEN
        return self.id2word[id]

    def size(self):
        return self.count


if __name__ == '__main__':
    # 读取文件路径
    rc = ReadConfig()
    raw_train_set = rc.get_path('raw_train_set')
    raw_test_set = rc.get_path('raw_test_set')
    train_set_x = rc.get_path('train_set_x')
    train_set_y = rc.get_path('train_set_y')
    test_set_x = rc.get_path('test_set_x')
    stop_words = rc.get_path('stop_words')
    vocab_path = rc.get_path('vocab_path')
    sentence_path = rc.get_path('sentence_path')
    w2v_model = rc.get_path('w2v_model')
    ft_model = rc.get_path('ft_model')
    w2v_vocab_metric = rc.get_path('w2v_vocab_metric')
    ft_vocab_metric = rc.get_path('ft_vocab_metric')

    # # step 1 清洗数据
    # train_x, train_y, test_x, _ = dc.parse_data(raw_train_set, raw_test_set)
    # dc.save_data(train_x, train_y, test_x, train_set_x, train_set_y, test_set_x, stop_words_path=stop_words)
    #
    # # step 2 构建需要的词表
    # lines = vb.read_data(train_set_x, train_set_y, test_set_x)
    # vocab, reverse_vocab = vb.build_vocab(lines)
    # vb.save_word_dict(vocab, vocab_path)
    #
    # # step 3  构建词向量,这里是已经对应了index转化之后的词向量
    # wb.build_by_word2vec(train_set_x, train_set_y, test_set_x, sentence_out_path=sentence_path)
    w2v_metric = wb.create_embedding_metric(vocab_path, w2v_model)
    print(w2v_metric[0])
    wb.dump_pkl(w2v_metric, w2v_vocab_metric)