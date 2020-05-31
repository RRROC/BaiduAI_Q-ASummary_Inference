import tensorflow as tf
from PGN_tf2.models.PGN import PGN
from utils.batcher_utils import batcher
from utils.embedding import Vocab
from PGN_tf2.helpers.train_helper import train_model


def train(params):
    global checkpoint_dir, ckpt, model
    assert params["mode"].lower() == "train", "change training mode to 'train'"
    assert params['model'] == "PGN", "change model to PGN to train"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    batch = batcher(vocab, params)
    print("Building the model ...")
    if params["model"] == "PGN":
        model = PGN(params)
    # elif params["model"] == "PGN":
    #     model = PGN(params)

    print("Creating the checkpoint manager")

    if params["model"] == "PGN":
        checkpoint_dir = "{}/checkpoint".format(params["PGN_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, batch, params, ckpt_manager, vocab)


if __name__ == '__main__':
    pass
