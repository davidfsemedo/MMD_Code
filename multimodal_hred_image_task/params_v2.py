import os
from tensorflow.contrib.rnn import GRUCell


def get_params(data_dir, dir, image_annoy_dir):
    """
    Creates a dictionary containing all the system parameters to be used in the v2 dataset
    :param data_dir: The path to the data folder
    :param dir: The path to output folder (Target_model)
    :param image_annoy_dir: The path to image_annoy (image_annoy_index)
    :return: The dictionary object containing all the system parameters
    """

    param = {}
    dir = str(dir)
    param['train_dir_loc'] = os.path.join(data_dir, "train/")
    param['valid_dir_loc'] = os.path.join(data_dir, "valid/")
    param['test_dir_loc'] = os.path.join(data_dir, "test/")
    param['dump_dir_loc'] = os.path.join(dir, "dump/")
    param['test_output_dir'] = os.path.join(dir, "test_output")
    param['vocab_file'] = os.path.join(dir, "vocab.pkl")
    param['image_annoy_dir'] = image_annoy_dir
    param['train_data_file'] = os.path.join(dir, "dump", "image", "train_data_file.pkl")
    param['valid_data_file'] = os.path.join(dir, "dump", "image", "valid_data_file.pkl")
    param['test_data_file'] = os.path.join(dir, "dump", "image", "test_data_file.pkl")
    param['vocab_file'] = os.path.join(dir, "vocab.pkl")
    param['vocab_stats_file'] = os.path.join(dir, "vocab_stats.pkl")
    param['model_path'] = os.path.join(dir, "model_image")
    param['terminal_op'] = os.path.join(dir, "terminal_output.txt")
    param['logs_path'] = os.path.join(dir, "log")
    param['text_embedding_size'] = 512
    param['image_rep_size'] = 4096
    param['image_embedding_size'] = 512
    param['activation'] = None  # tf.tanh
    param['output_activation'] = None  # tf.nn.softmax
    param['cell_size'] = 512
    param['cell_type'] = GRUCell
    param['batch_size'] = 64
    param['vocab_freq_cutoff'] = 2
    param['learning_rate'] = 0.0004
    param['patience'] = 200
    param['early_stop'] = 100
    param['max_epochs'] = 1000000
    param['max_len'] = 20
    param['max_negs'] = 5
    param['max_images'] = 1
    param['max_utter'] = 2 * param['max_negs']
    param['show_grad_freq'] = 20000
    param['valid_freq'] = 100
    param['print_train_freq'] = 1000
    param['max_gradient_norm'] = 0.1
    param['train_loss_incremenet_tolerance'] = 0.01
    return param
