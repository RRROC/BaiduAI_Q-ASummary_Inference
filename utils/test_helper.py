import tensorflow as tf


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    # print(enc_data)
    global outputs
    batch_data = enc_data[0]["enc_input"]
    batch_size = enc_data[0]["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data

    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
    # 这里解释下为什么要有一个batch_size,因为训练得时候是按照一个batch size扔进去得，所以得到得模型得输入结构也是如此，因此在测试得时候相当于将单个样本
    # 乘以batch size那么多遍，然后再得到结果，结果区list得第一个即可，当然理论上list得内容是一样得
    dec_input = tf.constant([vocab.word_to_id('[START]')] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    # print('enc_output shape is :',enc_output.get_shape())
    # print('dec_hidden shape is :', dec_hidden.get_shape())
    # print('inputs shape is :', inputs.get_shape())
    # print('dec_input shape is :', dec_input.get_shape())
    context_vector, _ = model.attention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

        # id转换
        predicted_ids = tf.argmax(predictions, axis=1).numpy()
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        # dec_input = tf.expand_dims(predicted_ids, 1)
    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results
