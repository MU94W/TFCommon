import tensorflow as tf
import os

class Model(object):
    """Base Model Class - for save and restore(load)
    """

    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name),
                        global_step=step)

    def load(self, save_path, model_file=None):
        assert os.path.exists(save_path), "[!] Checkpoints path does not exist..."
        print("[*] Reading checkpoints...")
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            assert ckpt and ckpt.model_checkpoint_path, "[!] No checkpoint file..."
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess,
                           os.path.join(save_path, ckpt_name))


