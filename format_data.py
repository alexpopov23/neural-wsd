import copy
import numpy as np
import tensorflow as tf


def get_one_hot_vector (position, size):
    vector = [0] * size
    vector[position] = 1
    return vector

def format_data(data_list, seq_width, src2id, lemma2synset, weights_biases):

    """
    :param data_list: list with the sentences to be used for training/testing
    :param seq_width: list with the lengths of the sentences (needed for the LSTM cell)
    :param src2id: mapping from source words to ids
    :param lemma2synset: mapping from lemmas to synset ids
    :param weights_biases: a dictionary mapping lemmas to their corresponding weight matrices and bias vectors for WSD
    :return: triple of lists containing the input words, expected output words and sequence lengths
    """

    input_data = np.empty([len(data_list), seq_width], dtype=int)
    labels = []
    weights = []
    biases = []
    seq_length = np.empty([len(data_list)], dtype=int)
    for count, sent in enumerate(data_list):
        if len(sent) > 50:
            sent = sent[:50]
        dummy_index = len(src2id)
        #TODO merge the two loops below into one -- computationally wasteful
        # Create a [seq_width, vocab_size]-shaped array, pad it with empty vectors when necessary.
        input_padded = [src2id[lemma] if lemma in src2id else src2id["UNK"] for _,lemma,_ in sent] \
                        + (seq_width - len(sent)) * [dummy_index]
        input_array = np.asarray(input_padded)
        input_data[count] = input_array
        labels_current = []
        weights_current = []
        biases_current = []
        for count, (_,lemma,synset) in enumerate(sent):
            if synset != "unspecified":

                if lemma in lemma2synset:
                    if lemma in src2id:
                        lemma_id = src2id[lemma]
                    else:
                        weights_current.append(weights_biases["w-unspecified"])
                        biases_current.append(weights_biases["b-unspecified"])
                        labels_current.append([0])
                        continue
                    synsets = lemma2synset[lemma]
                    syn_num = synsets.index(synset)
                    one_hot_vector = np.zeros([len(synsets)], dtype=int)
                    one_hot_vector[syn_num] = 1
                weights_current.append(weights_biases["w-"+str(lemma_id)])
                biases_current.append(weights_biases["b-"+str(lemma_id)])
                labels_current.append(one_hot_vector)
            else:
                weights_current.append(weights_biases["w-unspecified"])
                biases_current.append(weights_biases["b-unspecified"])
                #TODO swap with an array (just one - reference)
                labels_current.append([0])


        labels.append(labels_current)
        weights.append(weights_current)
        biases.append(biases_current)
        seq_length[count] = len(sent)
    return input_data, labels, seq_length, weights, biases

def format_data_selective(data_list, seq_width, src2id, lemma2synset):

    """
    :param data_list: list with the sentences to be used for training/testing
    :param seq_width: list with the lengths of the sentences (needed for the LSTM cell)
    :param src2id: mapping from source words to ids
    :param lemma2synset: mapping from lemmas to synset ids
    :param weights_biases: a dictionary mapping lemmas to their corresponding weight matrices and bias vectors for WSD
    :return: triple of lists containing the input words, expected output words and sequence lengths
    """

    input_data = np.empty([len(data_list), seq_width], dtype=int)
    labels = []
    #weights = []
    #biases = []
    weights_biases_id = []
    weights_biases_shape = []
    training_points = []
    seq_length = np.empty([len(data_list)], dtype=int)
    for count, sent in enumerate(data_list):
        if len(sent) > 50:
            sent = sent[:50]
        dummy_index = len(src2id)
        #TODO merge the two loops below into one -- computationally wasteful
        # Create a [seq_width, vocab_size]-shaped array, pad it with empty vectors when necessary.
        input_padded = [src2id[lemma] if lemma in src2id else src2id["UNK"] for _,lemma,_ in sent] \
                        + (seq_width - len(sent)) * [dummy_index]
        input_array = np.asarray(input_padded)
        input_data[count] = input_array
        labels_current = []
        #weights_current = []
        #biases_current = []
        weights_biases_id_current = []
        weights_biases_shape_current = []
        training_points_sent = []
        # token weight matrix and bias to attach to synsetless words
        token_matrix = np.zeros([200,1], dtype=float)
        token_bias = np.zeros([1], dtype=float)
        token_label = np.zeros([1], dtype=int)
        for count, (_,lemma,synset) in enumerate(sent):
            if synset != "unspecified":
                training_points_sent.append(count)
                one_hot_vector = [0]
                lemma_id = "unspecified"
                if lemma in lemma2synset:
                    if lemma in src2id:
                        lemma_id = src2id[lemma]
                    #else:
                    #    # TODO check if there are any such cases and figure out solution
                    #    weights_current.append(weights_biases["w-unspecified"])
                    #    biases_current.append(weights_biases["b-unspecified"])
                    #    labels_current.append([0])
                    #   continue
                    synsets = lemma2synset[lemma]
                    syn_num = synsets.index(synset)
                    one_hot_vector = np.zeros([len(synsets)], dtype=int)
                    one_hot_vector[syn_num] = 1
                one_hot_vector = np.asarray(one_hot_vector)
                #weights_current.append(weights_biases["w-"+str(lemma_id)])
                #biases_current.append(weights_biases["b-"+str(lemma_id)])
                weights_biases_id_current.append(str(lemma_id))
                weights_biases_shape_current.append(len(synsets))
                labels_current.append(one_hot_vector)
            else:
                #weights_current.append(token_matrix)
                #biases_current.append(token_bias)
                weights_biases_id_current.append(str(len(src2id)))
                weights_biases_shape_current.append(1)
                #TODO swap with an array (just one - reference)
                labels_current.append(token_label)
        labels_current += (seq_width - len(sent)) * [token_label]
        labels.append(labels_current)
        weights_biases_id_current += (seq_width - len(sent)) * ["0"]
        weights_biases_current = np.asarray(weights_biases_current)
        weights_biases_shape_current += (seq_width - len(sent)) * [1]
        weights_biases_shape_current = np.asarray(weights_biases_shape_current)
        # TODO attach shapes of weights/biases (for get_variable())
        weights_biases_id.append(weights_biases_current)
        weights_biases_shape.append(weights_biases_shape_current)
        #weights.append(weights_current)
        #biases_current += (seq_width - len(sent)) * [token_bias]
        #biases.append(biases_current)
        training_points.append(training_points_sent)
        seq_length[count] = len(sent)
    return input_data, labels, seq_length, weights_biases_id, weights_biases_shape, training_points