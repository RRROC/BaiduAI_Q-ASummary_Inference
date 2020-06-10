import tensorflow as tf
from PGN_tf2.encoders import encoder
from PGN_tf2.decoders.decoder import Decoder, Pointer, BahdanauAttention
from utils.data_utils import load_word2vec


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_word2vec(params, max_vocab_size=params["vocab_size"])
        self.params = params
        self.encoder = encoder.Encoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       enc_units=params["enc_units"],
                                       batch_size=params["batch_size"])

        self.attention = BahdanauAttention(units=params["attn_units"])

        self.decoder = Decoder(vocab_size=params["vocab_size"],
                               embedding_dim=params["embed_size"],
                               embedding_matrix=self.embedding_matrix,
                               dec_units=params["dec_units"],
                               batch_size=params["batch_size"])
        self.pointer = Pointer()

    def call_decoder_one_step(self, dec_input, dec_hidden, enc_output, enc_extended_inp, batch_oov_len,
                                  enc_pad_mask,
                                  use_coverage, prev_coverage):
        context_vector, attentions, coverage_ret = self.attention(dec_hidden,
                                                                  enc_output,
                                                                  enc_pad_mask,
                                                                  use_coverage,
                                                                  prev_coverage)
        dec_x, pred, dec_hidden = self.decoder(dec_input,
                                               dec_hidden,
                                               enc_output,
                                               context_vector)
        if self.params["model"] == 'PGN':
            p_gen = self.pointer(context_vector, dec_hidden, dec_x)
            final_dists = _calc_final_dist(enc_extended_inp,
                                           [pred],
                                           [tf.squeeze(attentions, axis=2)],
                                           [p_gen],
                                           batch_oov_len,
                                           self.params["vocab_size"],
                                           self.params["batch_size"])
            return tf.stack(final_dists, 1), dec_hidden, context_vector, tf.squeeze(attentions,
                                                                                    axis=2), p_gen, coverage_ret
        else:
            return pred, dec_hidden, context_vector, attentions, None, coverage_ret

    def call(self, dec_hidden, enc_output, dec_inp,
             enc_extended_inp, batch_oov_len,
             enc_pad_mask, use_coverage, prev_coverage=None):
        predictions = []
        attentions = []
        p_gens = []
        coverages = []

        context_vector, _, coverage_ret = self.attention(dec_hidden,
                                                         enc_output,
                                                         enc_pad_mask,
                                                         use_coverage,
                                                         prev_coverage)
        for t in range(dec_inp.shape[1]):
            # decoder
            # using teacher forcing
            dec_x, dec_pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                       dec_hidden,
                                                       enc_output,
                                                       context_vector)

            context_vector, attn, coverage_ret = self.attention(dec_hidden,
                                                                enc_output,
                                                                enc_pad_mask,
                                                                use_coverage,
                                                                coverage_ret)

            p_gen = self.pointer(context_vector, dec_hidden, dec_x)
            coverages.append(coverage_ret)
            attentions.append(tf.squeeze(attn, axis=2))
            predictions.append(dec_pred)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_inp,
                                       predictions,
                                       attentions,
                                       p_gens,
                                       batch_oov_len,
                                       self.params["vocab_size"],
                                       self.params["batch_size"])
        if self.params["mode"] == "train":
            return final_dists, dec_hidden, attentions, tf.stack(coverages, 1)
        else:
            return tf.stack(final_dists, 1), dec_hidden, attentions, tf.stack(coverages, 1)


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    # print('%'*30)
    # import numpy as  np
    # print(np.array(vocab_dists_extended).shape)
    # print(np.array(attn_dists_projected).shape)
    # print('%' * 30)
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists
