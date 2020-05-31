import tensorflow as tf
from seq2seq_tf2.encoders import encoder
from seq2seq_tf2.decoders import decoder
from utils.data_utils import load_word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        print(params["batch_size"])
        self.encoder = encoder.Encoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       enc_units=params["enc_units"],
                                       batch_size=params["batch_size"])

        self.attention = decoder.BahdanauAttention(units=params["attn_units"])

        self.decoder = decoder.Decoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       dec_units=params["dec_units"],
                                       batch_size=params["batch_size"])

    # def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
    #     # context_vector ()
    #     # attention_weights ()
    #     context_vector, attention_weights = self.attention(dec_hidden, enc_output)
    #
    #     # pred ()
    #     pred, dec_hidden = self.decoder(dec_input,
    #                                     None,
    #                                     None,
    #                                     context_vector)
    #     return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)

            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)
            # for i in range(dec_input.shape[0]):
            #     if dec_input[i][0] > self.params['vocab_size'] - 1:
            #         part1 = dec_input[:i]
            #         part2 = dec_input[i + 1:]
            #         val = tf.constant([[1]])
            #         dec_input = tf.concat([part1, val, part2], axis=0)

            predictions.append(pred)
            attentions.append(attn)
            # tf.concat与tf.stack这两个函数作用类似，
            # 都是在某个维度上对矩阵(向量）进行拼接，
            # 不同点在于前者拼接后的矩阵维度不变，后者则会增加一个维度。
        return tf.stack(predictions, 1), dec_hidden
