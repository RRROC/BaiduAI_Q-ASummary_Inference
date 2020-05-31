import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.use_bi_gru = False
        # 双向
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)

        initial_state = tf.zeros((self.batch_size, self.enc_units))

        if self.use_bi_gru:
            # 是否使用双向GRU
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded,
                                                                initial_state=[initial_state, initial_state])
            print(output)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)

        else:
            # 单向GRU
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)
            print(output)

        return output, enc_hidden
