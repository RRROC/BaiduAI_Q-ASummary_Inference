import tensorflow as tf
from seq2seq_tf2.models.seq2seq import SequenceToSequence
from utils.batcher_utils import batcher
from utils.embedding import Vocab
from seq2seq_tf2.helpers.test_helper import batch_greedy_decode
from seq2seq_tf2.helpers.test_helper import beam_decode
from tqdm import tqdm
import pandas as pd


def test(params):
    global model, ckpt, checkpoint_dir
    assert params['mode'].lower() == 'test', "change training mode to 'test' or 'eval'"

    print('Building the model....')
    if params['model'] == 'SequenceToSequence':
        model = SequenceToSequence(params)

    print('Creating vocab.....')
    vocab = Vocab(params['vocab_path'], params['vocab_size'])

    print('Creating the batcher...')
    b = batcher(vocab, params)

    print('Creating the checkpoint manager.......')
    if params['model'] == 'SequenceToSequence':
        checkpoint_dir = '{}/checkpoint'.format(params['seq2seq_model_dir'])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

        ckpt.restore(ckpt_manager.latest_checkpoint)

        print('Model restored')
        for batch in b:
            yield batch_greedy_decode(model, batch, vocab, params)


def test_and_save(params):
    assert params['test_save_dir'], "provide a dir where to save the results"

    gen = test(params)
    result = []
    with tqdm(total=params['nums_to_test'], position=0, leave=True) as pbar:
        for i in range(params['nums_to_test']):
            trail = next(gen)
            trail = list(map(lambda x: x.replace(" ", ""), trail))
            result.append(trail[0])
            pbar.update(1)

    save_predict_result(result, params)


def save_predict_result(result, params):
    test_df = pd.read_csv(params['test_df_dir'])
    test_df = test_df.loc[:9]
    test_df['Prediction'] = result

    test_df = test_df[['QID', 'Prediction']]

    test_df.to_csv('./seq2seq_tf2/output/test_result.csv')
