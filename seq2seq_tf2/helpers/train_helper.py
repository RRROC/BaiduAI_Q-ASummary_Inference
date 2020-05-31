import tensorflow as tf
import time
from seq2seq_tf2.models.losses import loss_function
import numpy as np


def train_model(model, dataset, params, ckpt_manager, vocab):
    print(vocab)
    start_index = vocab.word_to_id('<START>')
    pad_index = vocab.word_to_id('<PAD>')

    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'],
                                            epsilon=params['eps'])

    # @tf.function()
    def train_step(enc_inp, dec_tar, pad_index):
        with tf.GradientTape() as tape:
            # print('enc_inp shape is final for model :', enc_inp.get_shape())
            print(enc_inp)
            enc_output, enc_hidden = model.encoder(enc_inp)
            # 第一个decoder输入 开始标签
            # dec_input (batch_size, 1)
            # dec_input = tf.expand_dims([start_index], 1)
            dec_input = tf.expand_dims([start_index] * params["batch_size"], 1)
            dec_hidden = enc_hidden
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_tar)
            loss = loss_function(dec_tar, predictions, pad_index)

        # 下面这三行是固定写法
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    for epoch in range(params['epochs']):
        t0 = time.time()
        step = 0
        total_loss = 0
        # print(len(dataset.take(params['steps_per_epoch'])))
        for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
            # 讲设你的样本数是1000，batch size10,一个epoch，我们一共有100次，200， 500， 40，20.
            batch_loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                                    batch[1]["dec_target"], pad_index)  # shape=(16, 50)
            total_loss += batch_loss
            step += 1
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, batch_loss.numpy()))

        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path,
                                                                              total_loss / step))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))
            # 学习率的衰减，按照训练的次数来更新学习率（tf1.x）
            lr = params['learning_rate'] * np.power(0.9, epoch + 1)
            optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            print("learning_rate=", optimizer.get_config()["learning_rate"])
