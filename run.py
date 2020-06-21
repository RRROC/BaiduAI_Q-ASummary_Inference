import sys
import os
import tensorflow as tf
import argparse
import seq2seq_tf2.training as seq2seq_training
import seq2seq_tf2.testing as seq2seq_testing
import PGN_tf2.training as PGN_training
import PGN_tf2.testing as PGN_testing
import pathlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# from testing import test_and_save

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent


def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=100,
                        help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30,
                        help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--beam_size", default=6,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)

    parser.add_argument("--vocab_size", default=30000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.05, help="Learning rate", type=float)
    parser.add_argument('--cov_loss_wt', default=1.0, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument('--eps', default=1e-12, help='eps', type=float)
    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)
    # path
    # /ckpt/checkpoint/checkpoint
    print(BASE_DIR)
    parser.add_argument("--seq2seq_model_dir", default='./seq2seq_tf2/ckpt/seq2seq', help="Model folder")
    parser.add_argument("--PGN_model_dir", default='./PGN_tf2/ckpt/PGN', help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default='./resource/output/train_set_x.txt',
                        help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default='./resource/output/train_set_y.txt',
                        help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default='./resource/output/test_set_x.txt',
                        help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default='./resource/output/vocab.txt', help="Vocab path")
    parser.add_argument("--word2vec_output", default='./resource/output/w2v_vocab_metric.txt',
                        help="Vocab path")
    parser.add_argument("--test_save_dir", default='./resource/output/', help="test_save_dir")
    parser.add_argument("--test_df_dir", default='./resource/input/AutoMaster_TestSet.csv')

    # others
    parser.add_argument("--steps_per_epoch", default=300, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--nums_to_test", default=10, help="Number of examples to test", type=int)
    parser.add_argument("--epochs", default=15, help="train epochs", type=int)

    # mode
    parser.add_argument("--mode", default='test', help="training, eval or test options")
    parser.add_argument("--model", default='PGN', help="which model to be slected")
    parser.add_argument("--use_coverage", default=True, help="is_coverage")
    parser.add_argument("--greedy_decode", default=False, help="greedy_decoder")
    parser.add_argument("--transformer", default=False, help="transformer")
    parser.add_argument("--use_GPU", default=True, help="transformer")

    args = parser.parse_args()
    params = vars(args)

    if params['use_GPU']:
        print('*******Using GPU**************')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    else:
        print('*******Using CPU**************')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if params['model'] == 'SequenceToSequence':
        if params["mode"] == "train":
            print('Using Seq2Seq to train...')
            seq2seq_training.train(params)
        elif params["mode"] == "test":
            print('Using Seq2Seq to test...')
            seq2seq_testing.test_and_save(params)
            pass

    elif params['model'] == 'PGN':
        if params["mode"] == "train":
            print('Using PGN to train...')
            PGN_training.train(params)
        elif params["mode"] == "test":
            print('Using PGN to test...')
            PGN_testing.test_and_save(params)
            pass


if __name__ == '__main__':
    main()
