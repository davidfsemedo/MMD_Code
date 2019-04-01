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
    max_probs_index = []
    max_probs = []

    if true_op is not None:
        # true_op = true_op.tolist()
        true_op = np.asarray(true_op).T.tolist()
        true_op_prob = []
    i = 0

    for op in valid_op:
        sys.stdout.flush()
        max_index = np.argmax(op, axis=1)
        max_prob = np.max(op, axis=1)
        max_probs.append(max_prob)
        max_probs_index.append(max_index)
        true_op_prob.append([v_ij[t_ij] for v_ij, t_ij in zip(op, true_op)])
        # if true_op is not None:
        #
        #     i = i + 1
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
    print(word_indices)
    sentence_list = []
    for sent in word_indices:
        word_list = []
        for word_index in sent:
            word = vocab[word_index]
            word_list.append(word)
        sentence_list.append(" ".join(word_list))
    return sentence_list


def custom_map_id_to_word(word_indices, vocab):
    sentence_list = []
    word_list = []
    for word_index in word_indices:
        word = vocab[word_index]
        word_list.append(word)

    sentence_list.append(" ".join(word_list))
    return sentence_list


def encode_uterrance(param, vocab_dict):
    def pad_or_clip_utterance(utterance, param):
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

    utterance = "Hi there!".lower()
    unknown_word_id = 2
    df = collections.defaultdict(lambda: 0)
    binarized_text_context = []
    try:
        utterance_words = nltk.word_tokenize(utterance)
    except:
        utterance_words = utterance.split(' ')

    utterance_words = pad_or_clip_utterance(utterance_words, param)

    utterance_word_ids = []

    for word in utterance_words:
        word_id = vocab_dict.get(word, unknown_word_id)
        utterance_word_ids.append(word_id)

    unique_word_indices = set(utterance_word_ids)
    for word_id in unique_word_indices:
        df[word_id] += 1
    binarized_text_context.append(utterance_word_ids)

    binarized_text_context = binarized_text_context * 10  # adding 9 more examples for padding

    return np.array([binarized_text_context, [[''], [''], [''], [''], [''], [''], [''], [''], [''], ['']],
                     binarized_text_context[0]]).reshape(-1, 3)


def main():
    data_dir = '/nas/Datasets/mmd/v2'
    dump_dir = '/home/l.fischer/MMD_Code/Target_model'
    image_annoy_dir = '/home/l.fischer/MMD_Code/image_annoy_index'
    param = get_params(data_dir, dump_dir, image_annoy_dir)

    vocab = pkl.load(open(param['vocab_file'], "rb"))
    param['decoder_words'] = len(vocab)

    with tf.Graph().as_default():
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

        saver = tf.train.Saver()
        sess = tf.Session()

        # Restoring the variables as they were in the best model we got when training

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

        perform_prediction(sess, model, param, logits, losses, vocab)

        # test_batch_text, test_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(
        #     param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'],
        #     encoded_user_uterrance)
        #
        # predicted_sentence = []
        # feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_text_target, batch_decoder_input,
        #                          batch_text_weight, True)
        #
        # dec_op = sess.run([logits], feed_dict=feed_dict)
        # prediction = get_predicted_sentence(dec_op, None, vocab)
        # pred_sentence = prediction[0][0]
        # print(f"this one : {pred_sentence}")


def get_pred_loss(sess, model, encoded_user_utterance, param, logits, losses):
    test_batch_text, test_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(
        param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'],
        encoded_user_utterance)

    feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_text_target, batch_decoder_input,
                             batch_text_weight, True)

    loss, dec_op = sess.run([losses, logits], feed_dict=feed_dict)
    return dec_op, loss


def perform_prediction(sess, model, param, logits, losses, vocab):
    vocab_dict = read_vocab(param['vocab_file'])
    encoded_user_uterrance = encode_uterrance(param, vocab_dict)
    enconded_user_uterrance_target_ids = encoded_user_uterrance[0, 2]
    enconded_user_uterrance_target = custom_map_id_to_word(enconded_user_uterrance_target_ids, vocab)

    test_op, batch_loss = get_pred_loss(sess, model, encoded_user_uterrance, param, logits, losses)

    batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(test_op,
                                                                                             enconded_user_uterrance_target_ids,
                                                                                             vocab)
    print(batch_predicted_sentence)


if __name__ == "__main__":
    main()
