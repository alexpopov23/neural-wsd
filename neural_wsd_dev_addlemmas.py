import argparse
import sys
import pickle
import os
import collections
import random

import tensorflow as tf
import numpy as np

import data_ops_final_addlemmas
from gensim.models import KeyedVectors

from copy import copy
from sklearn.metrics.pairwise import cosine_similarity


class ModelSingleSoftmax:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, is_first, synset2id, word_embedding_dim, vocab_size,
                 batch_size, seq_width, n_hidden, n_hidden_layers,
                 val_inputs, val_input_lemmas, val_seq_lengths, val_flags, val_indices, val_labels,
                 lemma_embedding_dim, vocab_size_lemmas, link_embeddings=False):
        self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, word_embedding_dim])
        self.embeddings = tf.Variable(self.emb_placeholder)
        self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        self.emb_placeholder_lemmas = tf.placeholder(tf.float32, shape=[vocab_size_lemmas, lemma_embedding_dim])
        self.embeddings_lemmas = tf.Variable(self.emb_placeholder_lemmas)
        self.set_embeddings_lemmas = tf.assign(self.embeddings_lemmas, self.emb_placeholder_lemmas, validate_shape=False)
        # self.embeddings = tf.get_variable(name="W", shape=[len(src2id), word_embedding_dim], dtype=tf.float32,
        #                     initializer=tf.constant_initializer(word_embeddings), trainable=True)
        # weights and biases for the transorfmation after the RNN
        #TODO pick an initializer
        if link_embeddings:
            additional_neurons = 100
        else:
            additional_neurons = 0
        self.weights = tf.get_variable(name="softmax-w", shape=[2*n_hidden + additional_neurons, len(synset2id)], dtype=tf.float32)
        self.biases = tf.get_variable(name="softmax-b", shape=[len(synset2id)], dtype=tf.float32)
        if link_embeddings:
            self.weights_emb = tf.get_variable(name="transformemb-w", shape=[word_embedding_dim*2, additional_neurons], dtype=tf.float32)
            self.biases_emb = tf.get_variable(name="transformemb-b", shape=[additional_neurons], dtype=tf.float32)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_inputs_lemmas = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width])
        self.train_labels = tf.placeholder(tf.int32, shape=[None, len(synset2id)])
        self.train_indices = tf.placeholder(tf.int32, shape=[None])
        self.val_inputs = tf.constant(val_inputs, tf.int32)
        self.val_inputs_lemmas = tf.constant(val_input_lemmas, tf.int32)
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32)
        self.val_flags = tf.constant(val_flags, tf.bool)
        #self.val_labels = tf.constant(val_labels, tf.int32)
        self.place = tf.placeholder(tf.int32, shape=val_labels.shape)
        self.val_labels = tf.Variable(self.place)
        self.val_indices = tf.constant(val_indices, tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        reuse = None if is_first else True

        def embed_inputs (input_words, input_lemmas):

            embedded_inputs = tf.nn.embedding_lookup(self.embeddings, input_words)
            embedded_inputs_lemmas = tf.nn.embedding_lookup(self.embeddings_lemmas, input_lemmas)
            embedded_inputs = tf.concat([embedded_inputs, embedded_inputs_lemmas], 2)

            return embedded_inputs

        # create parameters for all word sense models
        #with tf.variable_scope("word-sense-models") as scope:
        #with tf.variable_scope(tf.get_variable_scope()) as scope:

        def biRNN_WSD (embedded_inputs, seq_lengths, indices, weights, biases, labels, is_training, keep_prob,
                       weights_emb, biases_emb):


            with tf.variable_scope(tf.get_variable_scope()) as scope:

                # Bidirectional recurrent neural network with LSTM cells
                initializer = tf.random_uniform_initializer(-1, 1)
                # with tf.variable_scope('forward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
                fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                if is_training:
                    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                fw_multicell = tf.contrib.rnn.MultiRNNCell([fw_cell] * n_hidden_layers)

                # with tf.variable_scope('backward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
                bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                if is_training:
                    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                bw_multicell = tf.contrib.rnn.MultiRNNCell([bw_cell] * n_hidden_layers)

                #embedded_inputs = tf.unstack(tf.transpose(embedded_inputs, [1, 0, 2]))
                #embedded_inputs = tf.transpose(embedded_inputs, [1, 0, 2])

                # Get the blstm cell output
                # rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_inputs, dtype="float32",
                #                                                  sequence_length=seq_lengths)
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs, dtype="float32",
                                                                 sequence_length=seq_lengths)

                rnn_outputs = tf.concat(rnn_outputs, 2)
                #rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
                #rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

                scope.reuse_variables()

                rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*n_hidden])
                target_outputs = tf.gather(rnn_outputs, indices)
                #connect input embeddings directly to the classification layer, in addition to RNN outputs
                if link_embeddings:
                    input_embeddings = tf.gather(tf.reshape(embedded_inputs, [-1, 2*word_embedding_dim]), indices)
                    input_embeddings = tf.reshape(input_embeddings, [-1, 2*word_embedding_dim])
                    input_embeddings = tf.matmul(input_embeddings, weights_emb) + biases_emb
                    target_outputs = tf.concat([target_outputs, input_embeddings], 1)
                logits = tf.matmul(target_outputs, weights) + biases
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cost = tf.reduce_mean(losses)

            return cost, logits

        embedded_inputs = embed_inputs(self.train_inputs, self.train_inputs_lemmas)
        if link_embeddings:
            weights_emb = self.weights_emb
            biases_emb = self.biases_emb
        else:
            weights_emb = None
            biases_emb = None
        self.cost, self.logits = biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                                           self.weights, self.biases, self.train_labels,
                                           True, self.keep_prob, weights_emb, biases_emb)
                                           #True, self.keep_prob, self.weights_emb, self.biases_emb)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        #self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        #self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.96, momentum=0.1).minimize(self.cost)
        #self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(self.cost)
        #scope.reuse_variables()
        embedded_inputs = embed_inputs(self.val_inputs, self.val_inputs_lemmas)
        tf.get_variable_scope().reuse_variables()
        _, self.val_logits = biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                                       self.weights, self.biases, self.val_labels,
                                       False, 1.0, weights_emb, biases_emb)
                                       #False, 1.0, self.weights_emb, self.biases_emb)

class ModelVectorSimilarity:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, is_first, synset2id, word_embedding_dim, vocab_size,
                 batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths, val_flags, val_indices, val_labels):
        self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, word_embedding_dim])
        self.embeddings = tf.Variable(self.emb_placeholder)
        self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        # self.embeddings = tf.get_variable(name="W", shape=[len(src2id), word_embedding_dim], dtype=tf.float32,
        #                     initializer=tf.constant_initializer(word_embeddings), trainable=True)
        # weights and biases for the transorfmation after the RNN
        #TODO pick an initializer
        self.weights = tf.get_variable(name="softmax-w", shape=[2*n_hidden, word_embedding_dim], dtype=tf.float32)
        #self.weights = tf.Variable(name="softmax-w", initial_value=tf.random_normal([2*n_hidden, word_embedding_dim]), dtype=tf.float32)
        self.biases = tf.get_variable(name="softmax-b", shape=[word_embedding_dim], dtype=tf.float32)
        #self.biases = tf.Variable(name="softmax-b", initial_value=tf.random_normal([word_embedding_dim]), dtype=tf.float32)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width])
        self.train_labels = tf.placeholder(tf.float32, shape=[None, word_embedding_dim])
        self.train_indices = tf.placeholder(tf.int32, shape=[None])
        self.val_inputs = tf.constant(val_inputs, tf.int32)
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32)
        self.val_flags = tf.constant(val_flags, tf.bool)
        #self.val_labels = tf.constant(val_labels, tf.float32)
        self.place = tf.placeholder(tf.float32, shape=val_labels.shape)
        self.val_labels = tf.Variable(self.place)
        self.val_indices = tf.constant(val_indices, tf.int32)

        reuse = None if is_first else True

        # create parameters for all word sense models
        with tf.variable_scope("word-sense-models") as scope:
            # Bidirectional recurrent neural network with LSTM cells
            initializer = tf.random_uniform_initializer(-1, 1)
            #with tf.variable_scope('forward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
            #with tf.variable_scope('backward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)

            def biRNN_WSD (inputs, seq_lengths, indices, embeddings, weights, biases, labels):

                embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs)
                #embedded_inputs = tf.unstack(tf.transpose(embedded_inputs, [1, 0, 2]))
                #embedded_inputs = tf.transpose(embedded_inputs, [1, 0, 2])

                # Get the blstm cell output
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_inputs, dtype="float32",
                                                                 sequence_length=seq_lengths)
                rnn_outputs = tf.concat(rnn_outputs, 2)
                #rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

                scope.reuse_variables()

                rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*n_hidden])
                target_outputs = tf.gather(rnn_outputs, indices)
                logits = tf.matmul(target_outputs, weights) + biases
                #losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                losses = (labels - logits) ** 2
                # with normalized vectors
                #losses = (tf.nn.l2_normalize(labels, dim=1) - tf.nn.l2_normalize(logits, dim=1)) ** 2
                #cost = tf.reduce_mean(losses)
                cost = tf.reduce_mean(losses)

                return cost, logits

            self.cost, self.logits = biRNN_WSD(self.train_inputs, self.train_seq_lengths, self.train_indices,
                                               self.embeddings, self.weights, self.biases, self.train_labels)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
            scope.reuse_variables()
            _, self.val_logits = biRNN_WSD(self.val_inputs, self.val_seq_lengths, self.val_indices,
                                           self.embeddings, self.weights, self.biases, self.val_labels)

def run_epoch(session, model, data, keep_prob, mode):

    feed_dict = {}
    if mode != "application":
        inputs = data[0]
        inputs_lemmas = data[1]
        seq_lengths = data[2]
        labels = data[3]
        words_to_disambiguate = data[4]
        indices = data[5]
        feed_dict = { model.train_inputs : inputs,
                      model.train_seq_lengths : seq_lengths,
                      model.train_model_flags : words_to_disambiguate,
                      model.train_indices : indices,
                      model.train_labels : labels,
                      model.keep_prob : keep_prob,
                      model.train_inputs_lemmas: inputs_lemmas}
    if mode == "train":
        ops = [model.train_op, model.cost, model.logits]
    elif mode == "val":
        ops = [model.train_op, model.cost, model.logits, model.val_logits]
    elif mode == "application":
        ops = [model.val_logits]
    fetches = session.run(ops, feed_dict=feed_dict)

    return fetches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural POS tagger.')
    parser.add_argument("-data_source", dest="data_source", required=False, default="default",
                        help="Which corpus are we using? Needed to determine how to read the data.")
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training run or an application run? Options: train, application")
    parser.add_argument('-wsd_method', dest='wsd_method', required=True, default="similarity",
                        help='Which method is used for the final, WSD step: similarity or fullsoftmax?')
    parser.add_argument('-word_embedding_method', dest='word_embedding_method', required=True, default="tensorflow",
                        help='Which method is used for loading the pretrained embeddings: tensorflow, gensim, glove?')
    parser.add_argument('-word_embedding_input', dest='word_embedding_input', required=False, default="wordform",
                        help='Are these embeddings of wordforms or lemmas (options are: wordform, lemma)?')
    parser.add_argument('-embeddings_load_script', dest='embeddings_load_script', required=False, default="None",
                        help='Path to the Python file that creates the word2vec object.')
    parser.add_argument('-word_embeddings_src_path', dest='word_embeddings_src_path', required=True,
                        help='The path to the pretrained model with the word embeddings (for the source language).')
    parser.add_argument('-lemma_embeddings_src_path', dest='lemma_embeddings_src_path', required=False,
                        help='The path to the pretrained model with the lemma embeddings (for the source language).')
    parser.add_argument('-word_embeddings_src_train_data', dest='word_embeddings_src_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the source language.')
    parser.add_argument('-word_embedding_dim', dest='word_embedding_dim', required=True,
                        help='Size of the word embedding vectors.')
    parser.add_argument('-lemma_embedding_dim', dest='lemma_embedding_dim', required=True,
                        help='Size of the lemma embedding vectors.')
    parser.add_argument('-word_embedding_case', dest='word_embedding_case', required=False, default="lowercase",
                        help='Are the word embeddings trained on lowercased or mixedcased text?')
    parser.add_argument('-sense_embeddings_src_path', dest='sense_embeddings_src_path', required=False, default="None",
                        help='If a path to sense embeddings is passed to the script, label generation is done using them.')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.3,
                        help='How fast should the network learn.')
    parser.add_argument('-training_iterations', dest='training_iters', required=False, default=10000,
                        help='How many iterations should the network train for.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=100,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-sequence_width', dest='seq_width', required=False, default=50,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-training_data', dest='training_data', required=True, default="brown",
                        help='The path to the gold corpus used for training/testing.')
    parser.add_argument('-data_partition', dest='partition_point', required=False, default="0.9",
                        help='Where to take the test data from, if using just one corpus (SemCor).')
    parser.add_argument('-test_data', dest='test_data', required=False, default="None",
                        help='The path to the gold corpus used for testing.')
    parser.add_argument('-lexicon_mode', dest='lexicon_mode', required=False, default="full dictionary",
                        help='Whether to use a lexicon or only the senses attested in the corpora: *full_dictionary* or *attested_senses*.')
    parser.add_argument('-lexicon', dest='lexicon', required=False, default="None",
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-save_path', dest='save_path', required=False, default="None",
                        help='Path to where the model should be saved.')
    parser.add_argument('-keep_prob', dest='keep_prob', required=False, default="1",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-dropword', dest='dropword', required=False, default="0",
                        help='The probability of keeping an input word (dropword)')
    parser.add_argument('-link_embeddings', dest='link_embeddings', required=False,
                        help='Boolean parameter that indicates whether to connect input embeddings to classification layer.')

    # read the parameters for the model and the data
    args = parser.parse_args()
    data_source = args.data_source
    mode = args.mode
    wsd_method = args.wsd_method
    sense_embeddings_path = args.sense_embeddings_src_path
    word_embeddings_src_path = args.word_embeddings_src_path
    lemma_embeddings_src_path = args.lemma_embeddings_src_path
    word_embedding_method = args.word_embedding_method
    word_embedding_dim = int(args.word_embedding_dim)
    lemma_embedding_dim = int(args.lemma_embedding_dim)
    word_embedding_case = args.word_embedding_case
    word_embedding_input = args.word_embedding_input
    word_embeddings = {}
    src2id = {}
    id2src = {}
    id2src_lemmas = {}
    src2id_lemmas = {}
    # TODO: write separate functions for each way of loading and put those in data_ops
    if word_embedding_method == "gensim":
        word_embeddings_model = KeyedVectors.load_word2vec_format(word_embeddings_src_path, binary=False)
        word_embeddings = word_embeddings_model.syn0
        id2src = word_embeddings_model.index2word
        for i, word in enumerate(id2src):
            src2id[word] = i
    elif word_embedding_method == "tensorflow":
        embeddings_load_script = args.embeddings_load_script
        sys.path.insert(0, embeddings_load_script)
        import word2vec_optimized as w2v
        word_embeddings = {} # store the normalized embeddings; keys are integers (0 to n)
        #TODO load the vectors from a saved structure, this TF graph below is pointless
        with tf.Graph().as_default(), tf.Session() as session:
            opts = w2v.Options()
            opts.train_data = args.word_embeddings_src_train_data
            opts.save_path = word_embeddings_src_path
            opts.emb_dim = word_embedding_dim
            model = w2v.Word2Vec(opts, session)
            ckpt = tf.train.get_checkpoint_state(args.word_embeddings_src_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint to reload a model was found!")
            src2id = model._word2id
            id2src = model._id2word
            word_embeddings = session.run(model._w_in)
            word_embeddings = tf.nn.l2_normalize(word_embeddings, 1).eval()
            #word_embeddings = np.vstack((word_embeddings, word_embedding_dim * [0.0]))
    elif word_embedding_method == "glove":
        word_embeddings, src2id, id2src = data_ops_final_addlemmas.loadGloveModel(word_embeddings_src_path)
        word_embeddings = np.asarray(word_embeddings)
        src2id["UNK"] = src2id["unk"]
        del src2id["unk"]
    if "UNK" not in src2id:
        #TODO use a random distribution rather
        unk = np.zeros(word_embedding_dim)
        src2id["UNK"] = len(src2id)
        word_embeddings = np.concatenate((word_embeddings, [unk]))

    if lemma_embeddings_src_path != None:
        lemma_embeddings_model = KeyedVectors.load_word2vec_format(lemma_embeddings_src_path, binary=False)
        lemma_embeddings = lemma_embeddings_model.syn0
        id2src_lemmas = lemma_embeddings_model.index2word
        for i, word in enumerate(id2src_lemmas):
            src2id_lemmas[word] = i
        if "UNK" not in src2id_lemmas:
            # TODO use a random distribution rather
            unk = np.zeros(lemma_embedding_dim)
            src2id_lemmas["UNK"] = len(src2id_lemmas)
            lemma_embeddings = np.concatenate((lemma_embeddings, [unk]))

    # Network Parameters
    learning_rate = float(args.learning_rate) # Update rate for the weights
    training_iters = int(args.training_iters) # Number of training steps
    batch_size = int(args.batch_size) # Number of sentences passed to the network in one batch
    seq_width = int(args.seq_width) # Max sentence length (longer sentences are cut to this length)
    n_hidden = int(args.n_hidden)
    n_hidden_layers = int(args.n_hidden_layers)# Number of features/neurons in the hidden layer
    embedding_size = word_embedding_dim
    vocab_size = len(src2id)
    lexicon_mode = args.lexicon_mode
    lexicon = args.lexicon
    partition_point = float(args.partition_point)
    keep_prob = float(args.keep_prob)
    dropword = float(args.dropword)
    link_embeddings = bool(args.link_embeddings)

    data = args.training_data
    known_lemmas = set()
    if data_source == "default":
        data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos = \
            data_ops_final_addlemmas.read_folder_semcor(data, lexicon_mode=lexicon_mode, f_lex=lexicon)
    elif data_source == "uniroma":
        data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos, known_lemmas, synset2freq = \
            data_ops_final_addlemmas.read_data_uniroma(data, lexicon_mode=lexicon_mode, f_lex=lexicon)
    #random.shuffle(data)
    #train_data = data[:partition]
    test_data = args.test_data
    if test_data == "None":
        partition = int(len(data) * partition_point)
        if partition_point < 0.90:
            val_data = data[partition:int(len(data) * (partition_point + 0.1))]
            train_data = data[:partition] + data[int(len(data) * (partition_point + 0.1)):]
        elif partition_point >= 0.90:
            train_data = data[:partition]
            val_data = data[partition:]
    else:
        train_data = data
        if data_source == "default":
            val_data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos = \
            data_ops_final_addlemmas.read_folder_semcor(test_data, lemma2synsets, lemma2id, synset2id, mode="test")
        elif data_source == "uniroma":
            val_data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos, known_lemmas, synset2freq = \
            data_ops_final_addlemmas.read_data_uniroma(test_data, lemma2synsets, lemma2id, synset2id, known_lemmas, synset2freq, mode="test")

    synset_train_cases = {}
    for sent in train_data:
        for word in sent:
            synset = word[2]
            if synset == "unspecified":
                continue
            if synset not in synset_train_cases:
                synset_train_cases[synset] = 1
            else:
                synset_train_cases[synset] += 1

    # get synset embeddings if a path to a model is passed
    if sense_embeddings_path != "None":
        label_mappings = {}
        sense_embeddings_model = KeyedVectors.load_word2vec_format(sense_embeddings_path, binary=False)
        sense_embeddings_full = sense_embeddings_model.syn0
        sense_embeddings = np.zeros(shape=(len(synset2id), 300), dtype=float)
        id2synset_embeddings = sense_embeddings_model.index2word
        #synset2id_embeddings = {}
        for i, synset in enumerate(id2synset_embeddings):
            if synset in synset2id:
                sense_embeddings[synset2id[synset]] = copy(sense_embeddings_full[i])

    val_inputs, val_inputs_lemmas, val_seq_lengths, val_labels, val_words_to_disambiguate, \
    val_indices, val_lemmas_to_disambiguate, val_synsets_gold = data_ops_final_addlemmas.format_data_dropword\
                                                    (wsd_method, val_data, src2id, src2id_lemmas, synset2id,
                                                    seq_width, word_embedding_case, word_embedding_input,
                                                     sense_embeddings, dropword=0)
    val_data = [val_inputs, val_seq_lengths, val_labels, val_words_to_disambiguate]

    # Function to calculate the accuracy on a batch of results and gold labels
    def accuracy(logits, labels, lemmas, synsets_gold, statistics=False):

        mistakes = {}
        successes = {}
        synsets = set()
        matching_cases = 0
        eval_cases = 0
        for i, logit in enumerate(logits):
            #pruned_logit = np.zeros([len(synset2id)])
            max = -10000
            max_id = -1
            #gold_synsets = [src2id[syn] for syn in synsets_gold[i]]
            #gold_pos = id2pos[gold_synsets[0]]
            gold_synsets = synsets_gold[i]
            gold_pos = gold_synsets[0].split("-")[1]
            lemma = lemmas[i]
            if lemma not in known_lemmas:
                if len(lemma2synsets[lemma]) == 1:
                    max_id = lemma2synsets[lemma][0]
                elif len(lemma2synsets[lemma]) > 1:
                    if synset2freq[lemma] > 0:
                        max_id = synset2freq[lemma]
                    else:
                        max_id = random.choice(lemma2synsets[lemma])
            else:
                for synset in lemma2synsets[lemma]:
                    id = synset2id[synset]
                    # make sure we only evaluate on synsets of the correct POS category
                    # if id2pos[id] != gold_pos:
                    #     continue
                    if synset.split("-")[1] != gold_pos:
                        continue
                    #pruned_logit[id] = logit[id]
                    if logit[id] > max:
                        max = logit[id]
                        max_id = synset
            #make sure there is at least one synset with a positive score
            # if max < 0:
            #     pruned_logit[max_id] = max * -1
            gold_synset = id2synset[np.argmax(labels[i])]
            # if np.argmax(pruned_logit) == np.argmax(labels[i]):
            if max_id in gold_synsets:
                matching_cases += 1
                if statistics:
                    if gold_synset not in successes:
                        successes[gold_synset] = 1
                    else:
                        successes[gold_synset] += 1
                    synsets.add(gold_synset)
            elif statistics:
                if gold_synset not in mistakes:
                    mistakes[gold_synset] = 1
                else:
                    mistakes[gold_synset] += 1
                synsets.add(gold_synset)
            eval_cases += 1

        if statistics:
            synsets_err_rate = {}
            for synset in synsets:
                n_mistakes = mistakes[synset] if synset in mistakes else 0.0
                n_successes = successes[synset] if synset in successes else 0.0
                synsets_err_rate[synset] = n_mistakes / float(n_mistakes + n_successes)
            synsets_err_rate = collections.OrderedDict(sorted(synsets_err_rate.iteritems(), key=lambda (k,v): (v,k)))
            f_stats = open(os.path.join(args.save_path, "corpus_statistics.txt"), "w")
            toWrite = "Synset\tError %\tN of mistakes\tN of correct guesses\tN of test cases\tN of training examples\n"
            for synset, error in synsets_err_rate.iteritems():
                n_mistakes = mistakes[synset] if synset in mistakes else 0
                n_successes = successes[synset] if synset in successes else 0
                toWrite += synset + "\t" + str(error * 100) + "%" \
                           + "\t" + str(n_mistakes) \
                           + "\t" + str(n_successes) \
                           + "\t" + str(n_mistakes + n_successes) \
                           + "\t" + (str(synset_train_cases[synset]) if synset in synset_train_cases else "0") + "\n"

            f_stats.write(toWrite)
            f_stats.close()

        return (100.0 * matching_cases) / eval_cases

    def accuracy_cosine_distance (logits, labels, lemmas, synsets_gold):

        matching_cases = 0
        eval_cases = 0
        for i, logit in enumerate(logits):
            lemma = lemmas[i]
            poss_synsets = lemma2synsets[lemma]
            best_fit = "None"
            max_similarity = 0.0
            for j, synset in enumerate(poss_synsets):
                syn_id = synset2id[synset]
                #cos_similarity = 1 - spatial.distance.cosine(logit, sense_embeddings[syn_id])
                cos_sim = cosine_similarity(logit.reshape(1,-1), sense_embeddings[syn_id].reshape(1,-1))[0][0]
                if cos_sim > max_similarity:
                    max_similarity = cos_sim
                    best_fit = syn_id
            if best_fit == synsets_gold[i]:
                matching_cases += 1
            eval_cases += 1

        return (100.0 * matching_cases) / eval_cases

    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset):

        batch = data[offset:(offset+batch_size)]
        inputs, inputs_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold = \
            data_ops_final_addlemmas.format_data_dropword(wsd_method, batch, src2id, src2id_lemmas, synset2id, seq_width,
                                             word_embedding_case, word_embedding_input, sense_embeddings, dropword)
        return inputs, inputs_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold

    model = None
    if wsd_method == "similarity":
        model = ModelVectorSimilarity(True, synset2id, word_embedding_dim, vocab_size, batch_size, seq_width,
                      n_hidden, val_inputs, val_seq_lengths, val_words_to_disambiguate, val_indices, val_labels)
    elif wsd_method == "fullsoftmax":
        model = ModelSingleSoftmax(True, synset2id, word_embedding_dim, vocab_size, batch_size, seq_width,
                                   n_hidden, n_hidden_layers, val_inputs, val_inputs_lemmas, val_seq_lengths,
                                   val_words_to_disambiguate, val_indices, val_labels, lemma_embedding_dim,
                                   len(src2id_lemmas), link_embeddings)
    session = tf.Session()
    saver = tf.train.Saver()
    #session.run(tf.global_variables_initializer())
    if mode == "application":
        saver.restore(session, os.path.join(args.save_path, "model.ckpt-25000"))
        # fetches = run_epoch(session, model, val_data, mode="application")
        # #lemma2synsets =
        # for i in range(len(fetches)):
        #     print "Input sentence is: ",
        #     for j in xrange(len(val_data[0][i])):
        #         print val_data[0][i][j][0] + " ",
        #     print "\n"
        #     #_predictions = session.run([predictions], feed_dict=feed_dict)[0]
        #     # _predictions = _predictions.eval()
        #     # print "Output sequence is: ",
        #     # for k in xrange(fetches[i]):
        #     #     # Print the N best candidates for each word
        #     #     # best_five = np.argsort(_predictions[k])[-5:]
        #     #     # for candidate in best_five:
        #     #     #    print id2target[candidate] + "|",
        #     #     # print "\n"
        #     #     # Print just the top scoring candidate for each word
        #     #     #print id2target[np.argmax(_predictions[k])] + " ",
        #     # print "\n"
        # exit()
    else:
        init = tf.initialize_all_variables()
        session.run(init, feed_dict={model.emb_placeholder: word_embeddings, model.emb_placeholder_lemmas: lemma_embeddings,
                                         model.place: val_labels})

    #session.run(model.set_embeddings, feed_dict={model.emb_placeholder: word_embeddings})

    batch_loss = 0
    for step in range(training_iters):
        offset = (step * batch_size) % (len(data) - batch_size)
        inputs, inputs_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, synsets_gold = new_batch(offset)
        if (len(labels) == 0):
            continue
        input_data = [inputs, inputs_lemmas, seq_lengths, labels, words_to_disambiguate, indices]

        if (step % 50 == 0):
            fetches = run_epoch(session, model, input_data, keep_prob, mode="val")
            if wsd_method == "similarity":
                print 'Minibatch accuracy: ' + str(accuracy_cosine_distance(fetches[2], labels, lemmas_to_disambiguate, synsets_gold))
                print 'Validation accuracy: ' + str(accuracy_cosine_distance(fetches[3], val_labels, val_lemmas_to_disambiguate, val_synsets_gold))
            elif wsd_method == "fullsoftmax":
                print 'Minibatch accuracy: ' + str(accuracy(fetches[2], labels, lemmas_to_disambiguate, synsets_gold))
                print 'Validation accuracy: ' + str(accuracy(fetches[3], val_labels, val_lemmas_to_disambiguate, val_synsets_gold))
            if step % 10000 == 0:
                accuracy(fetches[3], val_labels, val_lemmas_to_disambiguate, val_synsets_gold, statistics=True)
        else:
            fetches = run_epoch(session, model, input_data, keep_prob, mode="train")

        if (fetches[1] is not None):
            batch_loss += fetches[1]

        if (step % 100) == 0:
            print 'EPOCH: %d' % step
            print 'Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss/100.0)
            batch_loss = 0.0

        # if (args.save_path != "None" and step % 25000 == 0 and step != 0):
        #     saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=step)
        #     # with open(os.path.join(args.save_path, 'model-instance.pkl'), 'wb') as output:
        #     #     pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
        #     with open(os.path.join(args.save_path, 'lemma2synsets.pkl'), 'wb') as output:
        #         pickle.dump(lemma2synsets, output, pickle.HIGHEST_PROTOCOL)
        #     with open(os.path.join(args.save_path, 'lemma2id.pkl'), 'wb') as output:
        #         pickle.dump(lemma2id, output, pickle.HIGHEST_PROTOCOL)
        #     with open(os.path.join(args.save_path, 'synset2id.pkl'), 'wb') as output:
        #         pickle.dump(synset2id, output, pickle.HIGHEST_PROTOCOL)
        #     with open(os.path.join(args.save_path, 'id2synset.pkl'), 'wb') as output:
        #         pickle.dump(id2synset, output, pickle.HIGHEST_PROTOCOL)