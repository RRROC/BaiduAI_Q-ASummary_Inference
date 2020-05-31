import tensorflow as tf
from PGN_tf2.models.PGN import PGN
from utils.batcher_utils import batcher
from utils.embedding import Vocab
from PGN_tf2.helpers.test_helper import beam_decode
from tqdm import tqdm
import pandas as pd


def test(params):
    global model, ckpt, checkpoint_dir
    assert params['mode'].lower() == 'test', "change training mode to 'test' or 'eval'"
    assert params['beam_size'] == params['batch_size'], "Beam size must be same as batch_size"
    assert params['model'] == 'PGN', 'Please change the model to PGN'

    print('Building the model....')
    model = PGN(params)

    print('Creating vocab.....')
    vocab = Vocab(params['vocab_path'], params['vocab_size'])

    print('Creating the batcher...')
    batch = batcher(vocab, params)

    print('Creating the checkpoint manager.......')
    checkpoint_dir = '{}/checkpoint'.format(params['PGN_model_dir'])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print('Model restored')
    else:
        print('Initializing from scratch')

    result = []
    test_step = 0
    for b in batch:
        if test_step < 10:
            result.append(beam_decode(model, b, vocab, params))
            test_step += 1
        else:
            break
    return result


def test_and_save(params):
    assert params['test_save_dir'], "provide a dir where to save the results"

    result = test(params)
    # print("prediction result is: ", result)
    save_predict_result(result, params)


def save_predict_result(result, params):
    test_df = pd.read_csv(params['test_df_dir'])
    test_df = test_df.loc[:9]
    test_df['Prediction'] = result
    test_df = test_df[['QID', 'Prediction']]
    print('Prediction result: ', test_df)
    test_df.to_csv('./PGN_tf2/output/PGN_test_result.csv')
