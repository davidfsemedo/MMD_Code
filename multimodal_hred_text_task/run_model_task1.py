import os
import sys
import random
import logging
import math
import numpy as np
import os.path
import tensorflow as tf
import _pickle as pkl

# disable deprecation warnings
# issue: https://github.com/tensorflow/tensorflow/issues/27023
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

sys.path.append(os.getcwd())
from params_v2 import get_params
from read_data_task1 import get_batch_data, get_dialog_dict
from hierarchy_model_text import Hierarchical_seq_model_text


def feeding_dict(model, inputs_text, inputs_image, target_text, decoder_text_inputs, text_weights, feed_prev):
    """Create the feed dictionary to be fed to the model in the training and validation phases"""

    feed_dict = {}
    for encoder_text_input, input_text in zip(model.encoder_text_inputs, inputs_text):
        for encoder_text_input_i, input_text_i in zip(encoder_text_input, input_text):
            feed_dict[encoder_text_input_i] = input_text_i

    for encoder_img_input, input_image in zip(model.encoder_img_inputs, inputs_image):
        for encoder_img_input_i, input_image_i in zip(encoder_img_input, input_image):
            feed_dict[encoder_img_input_i] = input_image_i

    for model_target_text_i, target_text_i in zip(model.target_text, target_text):
        feed_dict[model_target_text_i] = target_text_i

    for model_decoder_text_input, decoder_text_input in zip(model.decoder_text_inputs, decoder_text_inputs):
        feed_dict[model_decoder_text_input] = decoder_text_input

    for model_text_weight, text_weight in zip(model.text_weights, text_weights):
        feed_dict[model_text_weight] = text_weight

    feed_dict[model.feed_previous] = feed_prev
    return feed_dict


def check_dir(param):
    """
    Checks whether model and logs directory exists, if not then creates both directories for saving best model and logs
    """

    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])

    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])


def run_training(param):
    """
    Performs the actual training of the model
    This function encapsulates many other functions involved in the training process of the model
    :param param: The dictionary object containing the system parameters for the model (obtained from params_v*.get_params())
    """

    def get_train_loss(model, batch_dict):
        """
        This method starts by obtaining the batch data necessary to feed to the model
        After obtaining the feed dictionary it feeds it to the model, running the losses and logits operations
        :param model: The model in which to run the operations
        :param batch_dict: A batch of the training data in which to extract the desired information to build a feed dictionary
        :return: The output of the loss and logits operations for the training set
        """

        # Retrieve the necessary variables from the batch data to construct the feed dictionary
        train_batch_text, train_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(
            param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'],
            batch_dict)

        # Constructing the feed dictionary to be fed to the model
        if epoch < 0:
            feed_dict = feeding_dict(model, train_batch_text, train_batch_image, batch_text_target, batch_decoder_input,
                                     batch_text_weight, False)
        else:
            feed_dict = feeding_dict(model, train_batch_text, train_batch_image, batch_text_target, batch_decoder_input,
                                     batch_text_weight, True)

        # Running the actual training process (train_op) and obtaining the losses (losses) and the embeddings (logits)
        loss, dec_op, _ = sess.run([losses, logits, train_op], feed_dict=feed_dict)
        return loss, dec_op

    def get_valid_loss(model, batch_dict):
        """
        This method starts by obtaining the batch data necessary to feed to the model
        After obtaining the feed dictionary it feeds it to the model, running the losses and logits operations
        :param model: The model in which to run the operations
        :param batch_dict: A batch of the validation data in which to extract the desired information to build a feed dictionary
        :return: The output of the loss and logits operations for the validation set
        """

        # Retrieve the necessary variables from the batch data to construct the feed dictionary
        valid_batch_text, valid_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(
            param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'],
            batch_dict)

        # Constructing the feed dictionary to be fed to the model
        feed_dict = feeding_dict(model, valid_batch_text, valid_batch_image, batch_text_target, batch_decoder_input,
                                 batch_text_weight, True)

        # Running the validation operation to obtain the losses and the embeddings obtained
        loss, dec_op = sess.run([losses, logits], feed_dict)
        return loss, dec_op

    def get_sum_batch_loss(batch_loss):
        """Sums up the entire loss values for a given batch"""
        return np.sum(np.asarray(batch_loss))

    def perform_training(model, batch_dict):
        """
        Calls get_train_loss where the training operation actually happens, also
        sums up the entire loss values for that batch
        """
        # Running the actual training process
        batch_train_loss, dec_op = get_train_loss(model, batch_dict)

        # Sum up all the losses obtained in this batch
        return get_sum_batch_loss(batch_train_loss)

    def perform_evaluation(model, batch_dict):
        """
        Calls get_valid_loss where the validation set operations actually happens,
        also sums up the entire loss values for the given batch
        """
        # Running the actual validation process
        batch_valid_loss, valid_op = get_valid_loss(model, batch_dict)

        # Sum up all the losses obtained in this batch
        return get_sum_batch_loss(batch_valid_loss)

    def evaluate(model, epoch, valid_data):
        """
        Performs the training procedure in the validation set every 10000 training steps
        """

        print('Validation Started')

        valid_loss = 0
        n_batches = int(math.ceil(float(len(valid_data)) / float(param['batch_size'])))

        for i in range(n_batches):

            # Print at every 10 batches
            if i % 10 == 0:
                print('Validating: Epoch {}, Batch {}'.format(epoch, i))

            # Obtain a batch from the validation data
            batch_dict = valid_data[i * param['batch_size']:(i + 1) * param['batch_size']]

            # Perform the validation process and sum up the losses for this batch
            sum_batch_loss = perform_evaluation(model, batch_dict)

            # Sum the losses from this batch with all the other batches so far
            valid_loss = valid_loss + sum_batch_loss

        # Remove the log handlers
        log = logging.getLogger()
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)

        return float(valid_loss) / float(len(valid_data))

    def load_pkl(filename):
        """Loads the given file with pickle"""
        with open(filename, 'rb') as f:
            file = pkl.load(f)
        return file

    # Loading the Training data
    train_data = load_pkl(param['train_data_file'])
    print(param['image_annoy_dir'])
    print('Train dialogue dataset loaded')
    sys.stdout.flush()

    # Loading the Validation data
    valid_data = load_pkl(param['valid_data_file'])
    print('Valid dialogue dataset loaded')
    sys.stdout.flush()

    # Loading the Vocabulary of the dataset
    vocab = load_pkl(param['vocab_file'])
    vocab_size = len(vocab)
    param['decoder_words'] = vocab_size

    print('writing terminal output to file')
    f_out = open(param['terminal_op'], 'w')

    check_dir(param)

    n_batches = int(math.ceil(float(len(train_data)) / float(param['batch_size'])))
    print('number of batches ', n_batches, 'len train data ', len(train_data), 'batch size', param['batch_size'])

    model_file = os.path.join(param['model_path'], "best_model")

    with tf.Graph().as_default():

        # Instantiating the model with the required parameters
        model = Hierarchical_seq_model_text('text', param['text_embedding_size'], param['image_embedding_size'],
                                            param['image_rep_size'], param['cell_size'], param['cell_type'],
                                            param['batch_size'], param['learning_rate'], param['max_len'],
                                            param['max_utter'], param['max_images'], param['patience'],
                                            param['decoder_words'], param['max_gradient_norm'], param['activation'],
                                            param['output_activation'])

        # Instantiate the placeholder variables the model expects to receive as input
        model.create_placeholder()
        logits = model.inference()
        losses = model.loss_task_text(logits)

        # The actual training operation
        train_op, gradients = model.train(losses)
        print("model created")
        sys.stdout.flush()

        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()

        tb_placeholder = tf.placeholder(tf.float16, shape=None)

        # Creating summary variables to write to tensorboard
        tb_training_loss = tf.compat.v1.summary.scalar('Training Loss', tb_placeholder)
        tb_validation_loss = tf.compat.v1.summary.scalar('Validation Loss', tb_placeholder)

        tb_writer = tf.compat.v1.summary.FileWriter('./tensorboard/')

        old_model_file = None

        # Restoring to previous model if it exists
        if len(os.listdir(param['model_path'])) > 0:
            old_model_file = None
            try:
                checkpoint_file_lines = open(param['model_path'] + '/checkpoint').readlines()
                for line in checkpoint_file_lines:
                    if line.startswith('model_checkpoint_path:'):
                        old_model_file = os.path.join(param['model_path'], line.split('"')[1])
            except:
                old_model_file = None
        else:
            old_model_file = None
        if old_model_file is not None:
            # Restoring to a previous best model if exists
            print("best model exists.. restoring from that point")
            saver.restore(sess, old_model_file)
        else:
            # If there's no old model then start from the beginning by initializing the variables
            print("initializing fresh variables")
            sess.run(init)

        best_valid_loss = float("inf")
        best_valid_epoch = 0

        # Printing TensorFlow variables
        all_var = tf.all_variables()
        print('printing all', len(all_var), ' TF variables:')
        for var in all_var:
            print(var.name, var.get_shape())

        # Starting the training stage
        print('Training Started')
        sys.stdout.flush()
        last_overall_avg_train_loss = None
        overall_step_count = 0

        for epoch in range(param['max_epochs']):
            random.shuffle(train_data)
            train_loss = 0
            for i in range(n_batches):

                # Print an update every 10 batches
                if i % 10 == 0:
                    print('Training: Epoch {}, Batch {}'.format(epoch, i))

                overall_step_count = overall_step_count + 1

                # Obtain a batch data from the training set
                train_batch_dict = train_data[i * param['batch_size']:(i + 1) * param['batch_size']]

                # Perform the actual training process
                sum_batch_loss = perform_training(model, train_batch_dict)

                avg_batch_loss = sum_batch_loss / float(param['batch_size'])

                # Sum up the entire loss values for every batch
                train_loss = train_loss + sum_batch_loss

                # Store average training loss in tensorboard
                if overall_step_count % param['tb_store_loss_freq'] == 0:
                    avg_train_loss = float(train_loss) / float(i + 1)

                    print('Avg Train Loss = {}'.format(avg_train_loss))
                    _loss = sess.run(tb_training_loss, feed_dict={tb_placeholder: avg_train_loss})
                    tb_writer.add_summary(_loss, overall_step_count)

                # Run Model Against Validation Data
                # The process is ran against the validation data every 10000 steps
                if overall_step_count > 0 and overall_step_count % param['valid_freq'] == 0:

                    # Perform the evaluation process
                    overall_avg_valid_loss = evaluate(model, epoch, valid_data)

                    # Add to Tensorboard
                    tb_writer.add_summary(summary=sess.run(fetches=tb_validation_loss,
                                                           feed_dict={tb_placeholder: overall_avg_valid_loss}),
                                          global_step=overall_step_count)

                    # Save the model if validation loss improves
                    if best_valid_loss > overall_avg_valid_loss:
                        saver.save(sess, model_file)
                        best_valid_loss = overall_avg_valid_loss

            overall_avg_train_loss = train_loss / float(len(train_data))

            if last_overall_avg_train_loss is not None and overall_avg_train_loss > last_overall_avg_train_loss:
                diff = overall_avg_train_loss - last_overall_avg_train_loss
                if diff > param['train_loss_incremenet_tolerance']:
                    print(
                        'WARNING: training loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' % (
                            overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance']))
                else:

                    print(
                        'WARNING: training loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' % (
                            overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance']))
            last_overall_avg_train_loss = overall_avg_train_loss
            sys.stdout.flush()
        print('Training over')
        print('Evaluating on test data')
    f_out.close()


def main():
    # The path to the location of the dataset
    data_dir = '/nas/Datasets/MMD/dataset/v2/'

    # The path to Target_model folder
    dump_dir = sys.argv[2].strip()

    image_annoy_dir = '../image_annoy_index/'

    # Obtain the system parameters present in params_v2.py
    param = get_params(data_dir, dump_dir, image_annoy_dir)

    # If the datafiles have already been created then simply use them
    if os.path.exists(param['train_data_file']) and os.path.exists(param['valid_data_file']) and os.path.exists(
            param['test_data_file']):

        print('dictionary already exists')
        sys.stdout.flush()

    else:

        # Create the datafiles and vocabulary file
        get_dialog_dict(param)

        print('dictionary formed')
        sys.stdout.flush()

    run_training(param)


if __name__ == "__main__":
    main()
