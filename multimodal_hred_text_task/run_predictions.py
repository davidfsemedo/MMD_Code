import os
import nltk
import collections
import pickle as pkl
import numpy as np
import tensorflow as tf
from params_v2 import *
from read_data_task1 import *
from hierarchy_model_text import *


def load_pkl(filename):
    with open(filename, 'rb') as f:
        file = pkl.load(f)
    return file


def read_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab_file = pkl.load(f)

    return {word: word_id for word_id, word in vocab_file.items()}


def feeding_dict(model, inputs_text, inputs_image, target_text, decoder_text_inputs, text_weights, feed_prev):
    """Creates the feeding dictionary to feed to the model"""

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


def get_predicted_sentence(valid_op, true_op, vocab):
    """
    Returns the predicted sentence and the probability of each word
    """

    max_probs_index = []
    max_probs = []

    if true_op is not None:
        true_op = np.asarray(true_op).T.tolist()
        true_op_prob = []

    for op in valid_op:
        sys.stdout.flush()
        max_index = np.argmax(op, axis=1)
        max_prob = np.max(op, axis=1)
        max_probs.append(max_prob)
        max_probs_index.append(max_index)
        true_op_prob.append([v_ij[t_ij] for v_ij, t_ij in zip(op, true_op)])

    max_probs_index = np.transpose(max_probs_index)
    max_probs = np.transpose(max_probs)

    if true_op is not None:
        true_op_prob = np.asarray(true_op_prob)
        true_op_prob = np.transpose(true_op_prob)
        if true_op_prob.shape[0] != max_probs.shape[0] and true_op_prob.shape[1] != max_probs.shape[1]:
            raise Exception('some problem shape of true_op_prob', true_op_prob.shape)

    # max_probs is of shape batch_size, max_len
    pred_sentence_list = map_id_to_word(max_probs_index, vocab)
    return pred_sentence_list, max_probs, true_op_prob


def map_id_to_word(word_indices, vocab):
    """Decodes the encoded utterance into it's sentence"""

    sentence_list = []
    for sent in word_indices:
        word_list = []
        for word_index in sent:
            word = vocab[word_index]
            word_list.append(word)
        sentence_list.append(" ".join(word_list))
    return sentence_list


def encode_uterrance(user_query, param, vocab_dict):
    """
    Encodes the user utterance returning a list of shape:
    [
        the encoding of the text,
        the encoding of the image (not used, filled with blank lists),
        the encoding of the target text (in this case, the same as the encoding of the text)
    ]
    """

    def pad_or_clip_utterance(utterance, param):
        """
        Inner function to help format the utterance, adding a start, end and pad token (if needed)
        Clipping it if it surpasses the max_len utterance size
        Padding it if its size is less then the max_len utterance size
        """
        start_word_symbol = '</s>'
        end_word_symbol = '</e>'
        pad_symbol = '<pad>'

        if len(utterance) > (param["max_len"] - 2):
            utterance = utterance[:(param["max_len"] - 2)]
            utterance.append(end_word_symbol)
            utterance.insert(0, start_word_symbol)

        elif len(utterance) < (param["max_len"] - 2):
            pad_length = param["max_len"] - 2 - len(utterance)
            utterance.append(end_word_symbol)
            utterance.insert(0, start_word_symbol)
            utterance = utterance + [pad_symbol] * pad_length

        else:
            utterance.append(end_word_symbol)
            utterance.insert(0, start_word_symbol)

        return utterance

    # Vocab words are in lower case, so should the utterance
    utterance = user_query.lower()

    # To count the number of out-of-vocabulary words
    unknown_word_id = 2

    # To store the encoding of the text
    text_encoding = []

    try:
        utterance_words = nltk.word_tokenize(utterance)
    except:
        utterance_words = utterance.split(' ')

    utterance_words = pad_or_clip_utterance(utterance_words, param)

    utterance_word_ids = []

    # Adding the word id's to the text_encoding
    for word in utterance_words:
        word_id = vocab_dict.get(word, unknown_word_id)
        utterance_word_ids.append(word_id)

    # The format the model is expecting to receive is a list containing 10 encoded sentences
    # Since we only have one sentence (the user_utterance) we add the 9 missing sentences as a copy of the one we have

    text_encoding.append(utterance_word_ids)

    # Padding the remaining 9 sentences as a copy of the one we have
    text_encoding = text_encoding * 10  # adding 9 more examples for padding

    # Since we're not using images just use a blank representation
    image_encoding = [['']] * 10

    # we need to reshape our matrix to a 1 row (-1 infers the number of rows) and 3 columns
    return np.array([text_encoding, image_encoding, text_encoding[0]]).reshape(-1, 3)


def get_prediction(user_query):
    """
    Gets the bot suggested response for the user_query sent from the client
    :param user_query: The query we want to get a response to
    :return: A sentence containing a start token (</s>), the actual response, and ending tokens (</e>)
    """

    # The location of the data file we want to use
    data_dir = '/nas/Datasets/mmd/v2'

    # The dump_dir is where the model will be stored as well as other files
    dump_dir = '/home/l.fischer/MMD_Code/Target_model'

    # Image annoy dir is not currently being used, only to fulfill the parameter
    image_annoy_dir = '/home/l.fischer/MMD_Code/image_annoy_index'

    # Gets the system parameters to send to our model
    param = get_params(data_dir, dump_dir, image_annoy_dir)

    # Load the vocab file as {word_id: word} in order to decode the user_query
    vocab = pkl.load(open(param['vocab_file'], "rb"))

    param['decoder_words'] = len(vocab)

    # For prediction we just want a batch size of 1
    param['batch_size'] = 1

    with tf.Graph().as_default():

        # Instantiate the model to be used with the correct parameters
        model = Hierarchical_seq_model_text('text', param['text_embedding_size'], param['image_embedding_size'],
                                            param['image_rep_size'], param['cell_size'], param['cell_type'],
                                            param['batch_size'], param['learning_rate'], param['max_len'],
                                            param['max_utter'], param['max_images'], param['patience'],
                                            param['decoder_words'], param['max_gradient_norm'], param['activation'],
                                            param['output_activation'])

        model.create_placeholder()
        logits = model.inference()
        losses = model.loss_task_text(logits)

        print("model created")

        # Restoring the variables as they were in the best model we got when training
        saver = tf.train.Saver()
        sess = tf.Session()

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
            print("best model exists.. restoring from that point")
            saver.restore(sess, old_model_file)

        # At this point we have the best model we got while training

        return perform_prediction(sess, model, param, logits, losses, vocab, user_query)


def perform_prediction(sess, model, param, logits, losses, vocab, user_query):
    """
    Encodes the user_query and retrieves the predicted sentence for that encoded query
    """

    # Read the vocab file as {word: word_id} in order to encode the user_query
    vocab_dict = read_vocab(param['vocab_file'])

    # Retrieve the encoded utterance
    # In order to be compliant with what the model expects as input
    # The encoded utterance is a list containing the text_encoding, the image_encoding,
    # and the target_text encoding (in this case is the same as the text_encoding)
    encoded_user_uterrance = encode_uterrance(user_query, param, vocab_dict)

    # Obtain just the target_text encoding
    enconded_user_uterrance_target_ids = encoded_user_uterrance[0, 2]

    # Running the model with the encoded input to obtain the prediction operation
    pred_op, _ = get_pred_loss(sess, model, encoded_user_uterrance, param, logits, losses)

    # Obtain a List containing the best sentence for the input query
    batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(pred_op,
                                                                                             enconded_user_uterrance_target_ids,
                                                                                             vocab)
    return batch_predicted_sentence[0]


def get_pred_loss(sess, model, encoded_user_utterance, param, logits, losses):
    """
    Runs the model with the given input in order to obtain a prediction operation
    """

    # Get the batched data to feed the model
    # In the prediction case the param["batch_size"] is set to 1
    # We still respect the format of batched data the model expects
    batched_text, batched_image, batched_text_target, batched_decoder_input, batched_text_weight = get_batch_data(
        param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'],
        encoded_user_utterance)

    # Create the feeding dictionary to feed to the model
    feed_dict = feeding_dict(model, batched_text, batched_image, batched_text_target, batched_decoder_input,
                             batched_text_weight, True)

    loss, pred_op = sess.run([losses, logits], feed_dict=feed_dict)
    return pred_op, loss
