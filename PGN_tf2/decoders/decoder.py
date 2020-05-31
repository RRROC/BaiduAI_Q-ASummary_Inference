import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        #  model.keras.layers.Dense

    def __call__(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
         calculate attention and coverage from dec_hidden enc_output and prev_coverage
         one dec_hidden(word) by one dec_hidden
         dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
         prev_coverage is [batch_sz, max_len_x, 1]
         dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
         output context_vector [batch_sz, enc_units] attention_weights & coverage [batch_sz, max_len_x, 1]
         """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights sha== (batch_size, max_length, 1)
            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)

            attention_weights = tf.nn.softmax(masked_score, axis=1)

            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights, coverage


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, dec_input, prev_dec_hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_input = self.embedding(dec_input)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # dec_input (batch_size, 1, embedding_dim + hidden_size)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        # passing the concatenated vector to the GRU
        dec_output, dec_hidden = self.gru(dec_input)

        # dec_output shape == (batch_size * 1, hidden_size)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)
        return dec_input, pred, dec_hidden


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        # change dec_inp_context to [batch_sz,embedding_dim+enc_units]
        dec_inp = tf.squeeze(dec_inp, axis=1)
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
