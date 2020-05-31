import tensorflow as tf
import time
from PGN_tf2.models.losses import calc_loss
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
    def train_step(enc_inp, extended_enc_input, max_oov_len,
                   dec_input, dec_target,
                   enc_pad_mask, padding_mask):
        with tf.GradientTape() as tape:
            # 逐个预测序列
            # encoder
            enc_output, enc_hidden = model.encoder(enc_inp)
            dec_hidden = enc_hidden

            final_dists, _, attentions, coverages = model(dec_hidden,
                                                          enc_output,
                                                          dec_input,
                                                          extended_enc_input,
                                                          max_oov_len,
                                                          enc_pad_mask=enc_pad_mask,
                                                          use_coverage=params['use_coverage'],
                                                          prev_coverage=None)

            batch_loss, log_loss, cov_loss = calc_loss(dec_target, final_dists, padding_mask, attentions,
                                                       params['cov_loss_wt'],
                                                       params['use_coverage'],
                                                       params['model'])
            print('Batch_Loss: {}, Log_Loss: {}, cov_loss: {}'.format(batch_loss, log_loss, cov_loss))
        variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                    model.attention.trainable_variables + model.pointer.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, log_loss, cov_loss

    # max_train_steps = params['max_train_steps']
    best_loss = 100
    for epoch in range(params['epochs']):
        start_time = time.time()
        step = 0
        total_loss = 0
        total_log_loss = 0
        total_cov_loss = 0

        for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
            enc_batch = batch[0]
            dec_batch = batch[1]
            print('Step: ', step)
            batch_loss, log_loss, cov_loss = train_step(enc_batch["enc_input"],
                                                        enc_batch["extended_enc_input"],
                                                        enc_batch["max_oov_len"],
                                                        dec_batch["dec_input"],
                                                        dec_batch["dec_target"],
                                                        enc_pad_mask=enc_batch["encoder_pad_mask"],
                                                        padding_mask=dec_batch["decoder_pad_mask"])
            total_loss += batch_loss.numpy()
            total_log_loss += log_loss.numpy()
            total_cov_loss += cov_loss

            total_loss = float(format(total_loss, '.4f'))
            total_log_loss = float(format(total_log_loss, '.4f'))
            print('total_loss: {}, total_log_loss: {}'.format(total_loss, total_log_loss))
            step += 1
            if step % 50 == 0:
                if params['use_coverage']:
                    print('Epoch {} Batch {} avg_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(epoch + 1,
                                                                                                     step,
                                                                                                     total_loss / step,
                                                                                                     total_log_loss / step,
                                                                                                     total_cov_loss / step))
                else:
                    print('Epoch {} Batch {} avg_loss {:.4f}'.format(epoch + 1, step, total_loss / step))

        if epoch % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))
            # 学习率的衰减，按照训练的次数来更新学习率（tf1.x）
            # lr = params['learning_rate'] * np.power(0.9, epoch + 1)
            # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            # print("learning_rate=", optimizer.get_config()["learning_rate"])
