import argparse
import sys
import pickle
import os
import collections
import random

import tensorflow as tf
import numpy as np

import data_ops
from gensim.models import KeyedVectors

from copy import copy
from sklearn.metrics.pairwise import cosine_similarity

pos_map = {"!": ".", "#": ".", "$": ".", "''": ".", "(": ".", ")": ".", ",": ".", "-LRB-": ".", "-RRB-": ".",
           ".": ".", ":": ".", "?": ".", "CC": "CONJ", "CD": "NUM", "CD|RB": "X", "DT": "DET", "EX": "DET",
           "FW": "X", "IN": "ADP", "IN|RP": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJRJR": "ADJ", "JJS": "ADJ",
           "JJ|RB": "ADJ", "JJ|VBG": "ADJ", "LS": "X", "MD": "VERB", "NN": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
           "NNS": "NOUN", "NN|NNS": "NOUN", "NN|SYM": "NOUN", "NN|VBG": "NOUN", "NP": "NOUN", "PDT": "DET",
           "POS": "PRT", "PRP": "PRON", "PRP$": "PRON", "PRP|VBP": "PRON", "PRT": "PRT", "RB": "ADV", "RBR": "ADV",
           "RBS": "ADV", "RB|RP": "ADV", "RB|VBG": "ADV", "RN": "X", "RP": "PRT", "SYM": "X", "TO": "PRT",
           "UH": "X", "VB": "VERB", "VBD": "VERB", "VBD|VBN": "VERB", "VBG": "VERB", "VBG|NN": "VERB",
           "VBN": "VERB", "VBP": "VERB", "VBP|TO": "VERB", "VBZ": "VERB", "VP": "VERB", "WDT": "DET", "WH": "X",
           "WP": "PRON", "WP$": "PRON", "WRB": "ADV", "``": "."}
pos_map_simple = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

class ModelSingleSoftmax:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, synset2id, word_embedding_dim, vocab_size,
                 batch_size, seq_width, n_hidden, n_hidden_layers,
                 val_inputs, val_input_lemmas, val_seq_lengths, val_flags, val_indices, val_labels,
                 lemma_embedding_dim, vocab_size_lemmas, wsd_classifier="True", pos_classifier="False", freq_classifier="False",
                 pos_classes=0, val_pos_labels=None, hypernym_classifier="False", hyp2id=None, val_hyp_labels=None,
                 val_hyp_indices=None, val_freq_labels=None, freq2id=None):
        self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, word_embedding_dim])
        self.embeddings = tf.Variable(self.emb_placeholder)
        self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        if vocab_size_lemmas > 0:
            self.emb_placeholder_lemmas = tf.placeholder(tf.float32, shape=[vocab_size_lemmas, lemma_embedding_dim])
            self.embeddings_lemmas = tf.Variable(self.emb_placeholder_lemmas)
            self.set_embeddings_lemmas = tf.assign(self.embeddings_lemmas, self.emb_placeholder_lemmas, validate_shape=False)
        #TODO pick an initializer
        if wsd_classifier == "True":
            self.weights = tf.get_variable(name="softmax-w", shape=[2*n_hidden, len(synset2id)], dtype=tf.float32)
            self.biases = tf.get_variable(name="softmax-b", shape=[len(synset2id)], dtype=tf.float32)
            self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, len(synset2id)])
            self.train_indices = tf.placeholder(tf.int32, shape=[None])
        else:
            self.weights = None
            self.biases = None
            self.train_model_flags = None
            self.train_labels = None
            self.train_indices = None
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_inputs_lemmas = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        if pos_classifier == "True":
            self.weights_pos = tf.get_variable(name="softmax_pos-w", shape=[2*n_hidden, pos_classes], dtype=tf.float32)
            self.biases_pos = tf.get_variable(name="softmax_pos-b", shape=[pos_classes], dtype=tf.float32)
            self.labels_pos = tf.placeholder(name="pos_labels", shape=[None, pos_classes], dtype=tf.int32)
            self.val_labels_pos = tf.constant(val_pos_labels, tf.int32)
        else:
            self.weights_pos = None
            self.biases_pos = None
            self.labels_pos = None
            self.val_labels_pos = None
        if hypernym_classifier == "True":
            self.weights_hyp = tf.get_variable(name="softmax_hyp-w", shape=[2*n_hidden, len(hyp2id)], dtype=tf.float32)
            self.biases_hyp = tf.get_variable(name="softmax_hyp-b", shape=[len(hyp2id)], dtype=tf.float32)
            self.labels_hyp = tf.placeholder(name="hyp_labels", shape=[None, len(hyp2id)], dtype=tf.int32)
            self.indices_hyp = tf.placeholder(name="hyp_indices", shape=[None], dtype=tf.int32)
            self.val_labels_hyp = tf.constant(val_hyp_labels, tf.int32)
            self.val_indices_hyp = tf.constant(val_hyp_indices, tf.int32)
        else:
            self.weights_hyp = None
            self.biases_hyp = None
            self.labels_hyp = None
            self.indices_hyp = None
            self.val_labels_hyp = None
            self.val_indices_hyp = None
        if freq_classifier == "True":
            self.weights_freq = tf.get_variable(name="softmax_freq-w", shape=[2*n_hidden, len(freq2id)], dtype=tf.float32)
            self.biases_freq = tf.get_variable(name="softmax_freq-b", shape=[len(freq2id)], dtype=tf.float32)
            self.labels_freq = tf.placeholder(name="freq_labels", shape=[None, len(freq2id)], dtype=tf.int32)
            self.val_labels_freq = tf.constant(val_freq_labels, tf.int32)
        else:
            self.weights_freq = None
            self.biases_freq = None
            self.labels_freq = None
            self.val_labels_freq = None
        self.val_inputs = tf.constant(val_inputs, tf.int32)
        if vocab_size_lemmas > 0:
            self.val_inputs_lemmas = tf.constant(val_input_lemmas, tf.int32)
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32)
        self.val_flags = tf.constant(val_flags, tf.bool)
        self.place = tf.placeholder(tf.int32, shape=val_labels.shape)
        self.val_labels = tf.Variable(self.place)
        self.val_indices = tf.constant(val_indices, tf.int32)
        if hypernym_classifier == "True":
            self.val_hyp_indices = tf.constant(val_hyp_indices, tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        def embed_inputs (input_words, input_lemmas=None):

            embedded_inputs = tf.nn.embedding_lookup(self.embeddings, input_words)
            if input_lemmas != None:
                embedded_inputs_lemmas = tf.nn.embedding_lookup(self.embeddings_lemmas, input_lemmas)
                embedded_inputs = tf.concat([embedded_inputs, embedded_inputs_lemmas], 2)

            return embedded_inputs

        def biRNN_WSD (embedded_inputs, seq_lengths, indices, weights, biases, labels, is_training, keep_prob,
                       pos_classifier="False", wsd_classifier="True", weights_pos=None, biases_pos=None, labels_pos=None,
                       hypernym_classifier="False", weights_hyp=None, biases_hyp=None, indices_hyp=None, labels_hyp=None,
                       weights_freq=None, biases_freq=None, labels_freq=None):

            with tf.variable_scope(tf.get_variable_scope()) as scope:

                # Bidirectional recurrent neural network with LSTM cells
                initializer = tf.random_uniform_initializer(-1, 1)
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
                def lstm_cell():
                    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                    if is_training:
                        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                    return lstm_cell

                # fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                # if is_training:
                #     fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                # fw_multicell = tf.contrib.rnn.MultiRNNCell([fw_cell] * n_hidden_layers)
                # # TODO: Use state_is_tuple=True
                # # TODO: add dropout
                # bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                # if is_training:
                #     bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                # bw_multicell = tf.contrib.rnn.MultiRNNCell([bw_cell] * n_hidden_layers)
                fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                # Get the blstm cell output
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs, dtype="float32",
                                                                 sequence_length=seq_lengths)
                rnn_outputs = tf.concat(rnn_outputs, 2)
                scope.reuse_variables()
                rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*n_hidden])
                logits_pos = []
                cost_pos = 0.0
                if pos_classifier == "True":
                    logits_pos = tf.matmul(rnn_outputs, weights_pos) + biases_pos
                    losses_pos = tf.nn.softmax_cross_entropy_with_logits(logits=logits_pos, labels=labels_pos)
                    cost_pos = tf.reduce_mean(losses_pos)
                logits = []
                losses = []
                cost_wsd = 0.0
                if wsd_classifier == "True":
                    target_outputs = tf.gather(rnn_outputs, indices)
                    logits = tf.matmul(target_outputs, weights) + biases
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cost_wsd = tf.reduce_mean(losses)
                logits_hyp = []
                cost_hyp = 0.0
                if hypernym_classifier == "True":
                    target_hyp_outputs = tf.gather(rnn_outputs, indices_hyp)
                    logits_hyp = tf.matmul(target_hyp_outputs, weights_hyp) + biases_hyp
                    losses_hyp = tf.nn.softmax_cross_entropy_with_logits(logits=logits_hyp, labels=labels_hyp)
                    cost_hyp = tf.reduce_mean(losses_hyp)
                cost_freq = 0.0
                if freq_classifier == "True":
                    logits_freq = tf.matmul(rnn_outputs, weights_freq) + biases_freq
                    losses_freq = tf.nn.softmax_cross_entropy_with_logits(logits=logits_freq, labels=labels_freq)
                    cost_freq = tf.reduce_mean(losses_freq)
                cost = cost_wsd + cost_pos + cost_hyp + cost_freq
                # if pos_classifier == "True" and wsd_classifier == "True":
                #     cost = cost_pos + cost_wsd
                # elif wsd_classifier == "True":
                #     cost = cost_wsd
                # elif pos_classifier == "True":
                #     cost = cost_pos

            return cost, logits, losses, logits_pos, logits_hyp, logits_freq

        # if lemma embeddings are passed, then concatenate them with the word embeddings
        if vocab_size_lemmas > 0:
            embedded_inputs = embed_inputs(self.train_inputs, self.train_inputs_lemmas)
        else:
            embedded_inputs = embed_inputs(self.train_inputs)
        self.cost, self.logits, self.losses, self.logits_pos, self.logits_hyp, self.logits_freq = \
                biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                        self.weights, self.biases, self.train_labels, True, self.keep_prob,
                        pos_classifier, wsd_classifier, self.weights_pos, self.biases_pos,
                        self.labels_pos, hypernym_classifier, self.weights_hyp, self.biases_hyp,
                        self.indices_hyp, self.labels_hyp, self.weights_freq, self.biases_freq, self.labels_freq)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        #self.train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.cost)
        if vocab_size_lemmas > 0:
            embedded_inputs = embed_inputs(self.val_inputs, self.val_inputs_lemmas)
        else:
            embedded_inputs = embed_inputs(self.val_inputs)
        tf.get_variable_scope().reuse_variables()
        _, self.val_logits, _, self.val_logits_pos, self.val_logits_hyp, self.val_logits_freq = \
            biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                      self.weights, self.biases, self.val_labels, False, 1.0,
                      pos_classifier, wsd_classifier, self.weights_pos, self.biases_pos, self.val_labels_pos,
                      hypernym_classifier, self.weights_hyp, self.biases_hyp, self.val_indices_hyp,
                      self.val_labels_hyp, self.weights_freq, self.biases_freq, self.val_labels_freq)

class ModelVectorSimilarity:

    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, input_mode, output_embedding_dim, lemma_embedding_dim, vocab_size_lemmas, batch_size, seq_width,
                 n_hidden, val_inputs, val_seq_lengths, val_flags, val_indices, val_labels, word_embedding_dim,
                 vocab_size_wordforms):

        if vocab_size_lemmas > 0:
            self.emb_placeholder_lemmas = tf.placeholder(tf.float32, shape=[vocab_size_lemmas, lemma_embedding_dim],
                                                         name="placeholder_for_lemma_embeddings")
            self.embeddings_lemmas = tf.Variable(self.emb_placeholder_lemmas, name="lemma_embeddings")
            self.set_embeddings_lemmas = tf.assign(self.embeddings_lemmas, self.emb_placeholder_lemmas,
                                                   validate_shape=False)
        # self.embeddings_lemmas = tf.nn.l2_normalize(self.embeddings_lemmas, 0)
        if vocab_size_wordforms > 0:
            self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size_wordforms, word_embedding_dim],
                                                  name="placeholder_for_word_embeddings")
            self.embeddings = tf.Variable(self.emb_placeholder, name="word_embeddings")
            self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        #TODO pick an initializer
        self.weights = tf.get_variable(name="w", shape=[2*n_hidden, output_embedding_dim], dtype=tf.float32)
        self.biases = tf.get_variable(name="b", shape=[output_embedding_dim], dtype=tf.float32)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width], name="train_inputs")
        self.train_inputs_lemmas = tf.placeholder(tf.int32, shape=[batch_size, seq_width], name="train_input_lemmas")
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="train_seq_lengths")
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width], name="train_model_flags")
        self.train_labels = tf.placeholder(tf.float32, shape=[None, lemma_embedding_dim], name="train_labels")
        self.train_indices = tf.placeholder(tf.int32, shape=[None], name="train_indices")
        if vocab_size > 0:
            self.val_inputs = tf.constant(val_inputs, tf.int32, name="val_inputs")
        if vocab_size_lemmas > 0:
            self.val_inputs_lemmas = tf.constant(val_input_lemmas, tf.int32, name="val_input_lemmas")
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32, name="val_seq_lengths")
        self.val_flags = tf.constant(val_flags, tf.bool, name="val_flags")
        self.place = tf.placeholder(tf.float32, shape=val_labels.shape)
        self.val_labels = tf.Variable(self.place, name="val_labels")
        self.val_indices = tf.constant(val_indices, tf.int32, name="val_indices")
        self.keep_prob = tf.placeholder(tf.float32)

        def embed_inputs (inputs, inputs_optional=None):

            if input_mode == "joint":
                embeddings1 = self.embeddings_lemmas
                embeddings2 = self.embeddings
            elif input_mode == "wordform":
                embeddings1 = self.embeddings
            elif input_mode == "lemma":
                embeddings1 = self.embeddings_lemmas
            embedded_inputs = tf.nn.embedding_lookup(embeddings1, inputs)
            if input_mode == "joint":
                embedded_inputs_wordforms = tf.nn.embedding_lookup(embeddings2, inputs_optional)
                embedded_inputs = tf.concat([embedded_inputs, embedded_inputs_wordforms], 2)

            return embedded_inputs

        def biRNN_WSD (embedded_inputs, seq_lengths, indices, weights, biases, labels, is_training, keep_prob=1.0):

            with tf.variable_scope(tf.get_variable_scope()) as scope:

                # Bidirectional recurrent neural network with LSTM cells
                initializer = tf.random_uniform_initializer(-1, 1)
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
                def lstm_cell():
                    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                    if is_training:
                        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                    return lstm_cell

                # fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                # if is_training:
                #     fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                # fw_multicell = tf.contrib.rnn.MultiRNNCell([fw_cell] * n_hidden_layers)
                # # TODO: Use state_is_tuple=True
                # # TODO: add dropout
                # bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                # if is_training:
                #     bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob,)
                # bw_multicell = tf.contrib.rnn.MultiRNNCell([bw_cell] * n_hidden_layers)
                fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                # Get the blstm cell output
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs, dtype="float32",
                                                                 sequence_length=seq_lengths)
                rnn_outputs = tf.concat(rnn_outputs, 2)
                scope.reuse_variables()
                rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*n_hidden])
                target_outputs = tf.gather(rnn_outputs, indices)
                output_emb = tf.matmul(target_outputs, weights) + biases
                losses = (labels - output_emb) ** 2
                # losses = (tf.nn.l2_normalize(labels, 0) - tf.nn.l2_normalize(output_emb, 0)) ** 2
                cost = tf.reduce_mean(losses)

            return cost, output_emb


        # if lemma embeddings are passed, then concatenate them with the word embeddings
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas, self.train_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.train_inputs)
        self.cost, self.logits = biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                                           self.weights, self.biases, self.train_labels, True, self.keep_prob)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas, self.val_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.val_inputs)
        tf.get_variable_scope().reuse_variables()
        _, self.val_logits = biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                                       self.weights, self.biases, self.val_labels, False)


class ModelMultiTaskLearning:

    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, input_mode, synID_mapping, output_embedding_dim, lemma_embedding_dim, vocab_size_lemmas,
                 batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths, val_flags, val_indices,
                 val_labels_classification, val_labels_regression, word_embedding_dim, vocab_size_wordforms,
                 freq_classifier="False", val_freq_labels=None, freq2id=None):

        if vocab_size_lemmas > 0:
            self.emb_placeholder_lemmas = tf.placeholder(tf.float32, shape=[vocab_size_lemmas, lemma_embedding_dim],
                                                         name="placeholder_for_lemma_embeddings")
            self.embeddings_lemmas = tf.Variable(self.emb_placeholder_lemmas, name="lemma_embeddings")
            self.set_embeddings_lemmas = tf.assign(self.embeddings_lemmas, self.emb_placeholder_lemmas,
                                                   validate_shape=False)
        if vocab_size_wordforms > 0:
            self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size_wordforms, word_embedding_dim],
                                                  name="placeholder_for_word_embeddings")
            self.embeddings = tf.Variable(self.emb_placeholder, name="word_embeddings")
            self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        #TODO pick an initializer
        self.weights_classification = tf.get_variable(name="w_classification", shape=[2*n_hidden, len(synID_mapping)], dtype=tf.float32)
        self.biases_classification = tf.get_variable(name="b_classification", shape=[len(synID_mapping)], dtype=tf.float32)
        self.weights_regression = tf.get_variable(name="w_regression", shape=[2*n_hidden, output_embedding_dim], dtype=tf.float32)
        self.biases_regression = tf.get_variable(name="b_regression", shape=[output_embedding_dim], dtype=tf.float32)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width], name="train_inputs")
        self.train_inputs_lemmas = tf.placeholder(tf.int32, shape=[batch_size, seq_width], name="train_input_lemmas")
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="train_seq_lengths")
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width], name="train_model_flags")
        self.train_labels_classification = tf.placeholder(tf.float32,
                                                          shape=[None, len(synID_mapping)],
                                                          name="train_labels_classification")
        self.train_labels_regression = tf.placeholder(tf.float32,
                                                          shape=[None, output_embedding_dim],
                                                          name="train_labels_regression")
        self.train_indices = tf.placeholder(tf.int32, shape=[None], name="train_indices")
        if vocab_size > 0:
            self.val_inputs = tf.constant(val_inputs, tf.int32, name="val_inputs")
        if vocab_size_lemmas > 0:
            self.val_inputs_lemmas = tf.constant(val_input_lemmas, tf.int32, name="val_input_lemmas")
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32, name="val_seq_lengths")
        self.val_flags = tf.constant(val_flags, tf.bool, name="val_flags")
        self.place_c = tf.placeholder(tf.float32, shape=val_labels_classification.shape)
        self.place_r = tf.placeholder(tf.float32, shape=val_labels_regression.shape)
        self.val_labels_classification = tf.Variable(self.place_c, name="val_labels_classification")
        self.val_labels_regression = tf.Variable(self.place_r, name="val_labels_regression")
        self.val_indices = tf.constant(val_indices, tf.int32, name="val_indices")
        self.keep_prob = tf.placeholder(tf.float32)
        if freq_classifier == "True":
            self.weights_freq = tf.get_variable(name="softmax_freq-w", shape=[2*n_hidden, len(freq2id)], dtype=tf.float32)
            self.biases_freq = tf.get_variable(name="softmax_freq-b", shape=[len(freq2id)], dtype=tf.float32)
            self.labels_freq = tf.placeholder(name="freq_labels", shape=[None, len(freq2id)], dtype=tf.int32)
            self.val_labels_freq = tf.constant(val_freq_labels, tf.int32)
        else:
            self.weights_freq = None
            self.biases_freq = None
            self.labels_freq = None
            self.val_labels_freq = None

        def embed_inputs (inputs, inputs_optional=None):

            if input_mode == "joint":
                embeddings1 = self.embeddings_lemmas
                embeddings2 = self.embeddings
            elif input_mode == "wordform":
                embeddings1 = self.embeddings
            elif input_mode == "lemma":
                embeddings1 = self.embeddings_lemmas
            embedded_inputs = tf.nn.embedding_lookup(embeddings1, inputs)
            if input_mode == "joint":
                embedded_inputs_wordforms = tf.nn.embedding_lookup(embeddings2, inputs_optional)
                embedded_inputs = tf.concat([embedded_inputs, embedded_inputs_wordforms], 2)

            return embedded_inputs

        def biRNN_WSD (embedded_inputs, seq_lengths, indices, weights_c, biases_c, weights_r, biases_r,
                       labels_c, labels_r, is_training, keep_prob, weights_freq, biases_freq, labels_freq):

            with tf.variable_scope(tf.get_variable_scope()) as scope:

                # Bidirectional recurrent neural network with LSTM cells
                initializer = tf.random_uniform_initializer(-1, 1)
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
                def lstm_cell():
                    lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                    if is_training:
                        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
                    return lstm_cell

                fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
                # Get the blstm cell output
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs, dtype="float32",
                                                                 sequence_length=seq_lengths)
                rnn_outputs = tf.concat(rnn_outputs, 2)
                scope.reuse_variables()
                rnn_outputs = tf.reshape(rnn_outputs, [-1, 2*n_hidden])
                target_outputs = tf.gather(rnn_outputs, indices)
                output_c = tf.matmul(target_outputs, weights_c) + biases_c
                losses_c = tf.nn.softmax_cross_entropy_with_logits(logits=output_c, labels=labels_c)
                cost_c = tf.reduce_mean(losses_c)
                output_r = tf.matmul(target_outputs, weights_r) + biases_r
                losses_r = (labels_r - output_r) ** 2
                cost_r = tf.reduce_mean(losses_r)
                cost_freq = 0.0
                logits_freq = []
                if freq_classifier == "True":
                    logits_freq = tf.matmul(rnn_outputs, weights_freq) + biases_freq
                    losses_freq = tf.nn.softmax_cross_entropy_with_logits(logits=logits_freq, labels=labels_freq)
                    cost_freq = tf.reduce_mean(losses_freq)
                cost = cost_c + cost_r + cost_freq

            return cost, cost_c, cost_r, output_c, output_r, logits_freq


        # if lemma embeddings are passed, then concatenate them with the word embeddings
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas, self.train_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.train_inputs)
        self.cost, self.cost_c, self.cost_r, self.logits, self.output_emb, self.logits_freq = \
            biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                      self.weights_classification, self.biases_classification,
                      self.weights_regression, self.biases_regression,
                      self.train_labels_classification, self.train_labels_regression,
                      True, self.keep_prob, self.weights_freq, self.biases_freq, self.labels_freq)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas, self.val_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.val_inputs)
        tf.get_variable_scope().reuse_variables()
        _, _, _, self.val_logits, self.val_output_emb, self.val_logits_freq = \
            biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                      self.weights_classification, self.biases_classification,
                      self.weights_regression, self.biases_regression,
                      self.val_labels_classification, self.val_labels_regression,
                      False, self.keep_prob, self.weights_freq, self.biases_freq, self.val_labels_freq)

def run_epoch(session, model, data, keep_prob, mode, multitask="False"):

    feed_dict = {}
    if mode != "application":
        inputs = data[0]
        input_lemmas = data[1]
        seq_lengths = data[2]
        labels = data[3]
        words_to_disambiguate = data[4]
        indices = data[5]
        labels_pos = data[6]
        labels_hyp = data[7]
        indices_hyp = data[8]
        labels_freq = data[9]
        feed_dict = { model.train_seq_lengths : seq_lengths,
                      model.keep_prob : keep_prob}
        if wsd_classifier == "True":
            if multitask == "True":
                feed_dict.update({model.train_labels_classification: labels[0]})
                feed_dict.update({model.train_labels_regression: labels[1]})
            else:
                feed_dict.update({model.train_labels: labels})
            feed_dict.update({model.train_model_flags : words_to_disambiguate, model.train_indices : indices})
        if pos_classifier == "True":
            feed_dict.update({model.labels_pos : labels_pos})
        if hypernym_classifier == "True":
            feed_dict.update({model.labels_hyp : labels_hyp, model.indices_hyp : indices_hyp})
        if freq_classifier == "True":
            feed_dict.update({model.labels_freq : labels_freq})
        if len(inputs) > 0:
            feed_dict.update({model.train_inputs: inputs})
        if len(input_lemmas) > 0:
            feed_dict.update({model.train_inputs_lemmas : input_lemmas})
    if mode == "train":
        if multitask == "True":
            ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.output_emb]
        else:
            ops = [model.train_op, model.cost, model.logits, model.logits_pos]
    elif mode == "val":
        if multitask == "True":
            ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.val_logits,
                   model.output_emb, model.val_output_emb, model.logits_freq, model.val_logits_freq]
        else:
            ops = [model.train_op, model.cost, model.logits, model.val_logits, model.logits_pos, model.val_logits_pos,
                   model.logits_hyp, model.val_logits_hyp]
    elif mode == "application":
        ops = [model.val_logits]
    fetches = session.run(ops, feed_dict=feed_dict)

    return fetches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural WSD tagger.')
    parser.add_argument("-data_source", dest="data_source", required=False, default="uniroma",
                        help="Which corpus are we using? Needed to determine how to read the data. Options: naf, uniroma")
    parser.add_argument("-diff_data_sources", dest="diff_data_sources", required=False, default="False",
                        help="In case we want to read train data from naf files and test data from uniroma files.")
    parser.add_argument("-sensekey2synset", dest="sensekey2synset", required=False,
                        help="Specify where the uniroma mappings are (by default in the training folder).")
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training run or an application run? Options: train, application")
    parser.add_argument('-wsd_method', dest='wsd_method', required=True, default="fullsoftmax",
                        help='Which method is used for the final, WSD step: similarity or fullsoftmax?')
    parser.add_argument('-wsd_classifier', dest='wsd_classifier', required=False, default="True",
                        help='Should the system learn to annotate for WSD as well?')
    parser.add_argument('-pos_classifier', dest='pos_classifier', required=False, default="False",
                        help='Should the system learn to annotate for POS as well?')
    parser.add_argument('-hypernym_classifier', dest='hypernym_classifier', required=False, default="False",
                        help='Should the system learn to annotate for WordNet hypernyms as well?')
    parser.add_argument('-frequency_classifier', dest='frequency_classifier', required=False, default="False",
                        help='Should the system learn to annotate the frequency bin class of words as well?')
    parser.add_argument('-hypernymy_rels', dest='hypernymy_rels', required=False, default="False",
                        help='Path to file with hypernymy hierarchy.')
    parser.add_argument('-word_embedding_method', dest='word_embedding_method', required=False, default="tensorflow",
                        help='Which method is used for loading the pretrained embeddings: tensorflow, gensim, glove?')
    parser.add_argument('-joint_embedding', dest='joint_embedding', required=False,
                        help='Whether lemmas and synsets are jointly embedded.')
    parser.add_argument('-word_embedding_input', dest='word_embedding_input', required=False, default="wordform",
                        help='Are these embeddings of wordforms or lemmas (options are: wordform, lemma, joint)?')
    parser.add_argument('-word_embedding_case', dest='word_embedding_case', required=False, default="lowercase",
                        help='Are the word embeddings trained on lowercased or mixedcased text? Options: lowercase, mixedcase')
    parser.add_argument('-embeddings_load_script', dest='embeddings_load_script', required=False, default="None",
                        help='Path to the Python file that creates the word2vec object (tensorflow model).')
    parser.add_argument('-word_embeddings_src_path', dest='word_embeddings_src_path', required=False,
                        help='The path to the pretrained model with the word embeddings.')
    parser.add_argument('-word_embeddings_src_train_data', dest='word_embeddings_src_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the source language (tensorflow model).')
    parser.add_argument('-word_embedding_dim', dest='word_embedding_dim', required=False, default="0",
                        help='Size of the word embedding vectors.')
    parser.add_argument('-lemma_embeddings_src_path', dest='lemma_embeddings_src_path', required=False,
                        help='The path to the pretrained model with the lemma embeddings.')
    parser.add_argument('-lemma_embedding_dim', dest='lemma_embedding_dim', required=False, default="0",
                        help='Size of the lemma embedding vectors.')
    parser.add_argument('-use_pos', dest='use_pos', required=False, default="False",
                        help='Whether to append POS information to lemmas prior to embedding them.')
    parser.add_argument('-sense_embeddings_src_path', dest='sense_embeddings_src_path', required=False, default="None",
                        help='If a path to sense embeddings is passed to the script, label generation is done using them.')
    parser.add_argument('-synset_mapping', dest='synset_mapping', required=False,
                        help='A mapping between the synset embedding IDs and WordNet, if such is necessary.')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.3,
                        help='How fast should the network learn.')
    parser.add_argument('-training_iterations', dest='training_iters', required=False, default=100000,
                        help='How many iterations should the network train for.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=100,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-sequence_width', dest='seq_width', required=False, default=50,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-keep_prob', dest='keep_prob', required=False, default="1",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-dropword', dest='dropword', required=False, default="0",
                        help='The probability of keeping an input word (dropword)')
    parser.add_argument('-training_data', dest='training_data', required=True, default="brown",
                        help='The path to the gold corpus used for training/testing.')
    parser.add_argument('-data_partition', dest='partition_point', required=False, default="0.9",
                        help='Where to take the test data from, if using just one corpus (SemCor).')
    parser.add_argument('-test_data', dest='test_data', required=False, default="None",
                        help='The path to the gold corpus used for testing.')
    parser.add_argument('-lexicon', dest='lexicon', required=False, default="None",
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-lexicon_mode', dest='lexicon_mode', required=False, default="full_dictionary",
                        help='Whether to use a lexicon or only the senses attested in the corpora: *full_dictionary* or *attested_senses*.')
    parser.add_argument('-save_path', dest='save_path', required=False, default="None",
                        help='Path to where the model should be saved.')


    # Read the parameters for the model and the data
    args = parser.parse_args()
    data_source = args.data_source
    mode = args.mode
    wsd_method = args.wsd_method
    joint_embedding = args.joint_embedding
    if wsd_method == "multitask":
        multitask = "True"
    else:
        multitask = "False"
    word_embeddings_src_path = args.word_embeddings_src_path
    lemma_embeddings_src_path = args.lemma_embeddings_src_path
    sense_embeddings_src_path = args.sense_embeddings_src_path
    synset_mapping = args.synset_mapping
    word_embedding_method = args.word_embedding_method
    word_embedding_dim = int(args.word_embedding_dim)
    lemma_embedding_dim = int(args.lemma_embedding_dim)
    word_embedding_case = args.word_embedding_case
    word_embedding_input = args.word_embedding_input
    word_embeddings = {}
    lemma_embeddings = {}
    src2id = {}
    id2src = {}
    id2src_lemmas = {}
    src2id_lemmas = {}
    if word_embeddings_src_path != None:
        if word_embedding_method == "gensim":
            word_embeddings_model = KeyedVectors.load_word2vec_format(word_embeddings_src_path, binary=True)
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
            word_embeddings, src2id, id2src = data_ops.loadGloveModel(word_embeddings_src_path)
            word_embeddings = np.asarray(word_embeddings)
            src2id["UNK"] = src2id["unk"]
            del src2id["unk"]
        if src2id != None and "UNK" not in src2id:
            #TODO use a random distribution rather
            unk = np.zeros(word_embedding_dim)
            src2id["UNK"] = len(src2id)
            word_embeddings = np.concatenate((word_embeddings, [unk]))

    # change this to turn off/on using WSD-modified word vectors
    modified_embeddings = False
    if lemma_embeddings_src_path != None:
        if modified_embeddings:
            files = os.listdir(lemma_embeddings_src_path)
            for file in files:
                if file.startswith("embeddings"):
                    lemma_embeddings = pickle.load(open(os.path.join(lemma_embeddings_src_path, file), "rb"))
                elif file.startswith("src2id"):
                    src2id_lemmas = pickle.load(open(os.path.join(lemma_embeddings_src_path, file), "rb"))
        else:
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
    n_hidden_layers = int(args.n_hidden_layers) # Number of features/neurons in the hidden layer
    embedding_size = word_embedding_dim
    vocab_size = len(src2id)
    vocab_size_lemmas = len(src2id_lemmas)
    lexicon_mode = args.lexicon_mode
    lexicon = args.lexicon
    partition_point = float(args.partition_point)
    keep_prob = float(args.keep_prob)
    dropword = float(args.dropword)
    use_pos = args.use_pos
    pos_classifier = args.pos_classifier
    wsd_classifier = args.wsd_classifier
    hypernym_classifier = args.hypernym_classifier
    freq_classifier = args.frequency_classifier
    hypernymy_rels = args.hypernymy_rels
    diff_data_sources = args.diff_data_sources
    sensekey2synset = args.sensekey2synset

    if hypernym_classifier == "True":
        hyponyms, syn2hyp = data_ops.get_hypernymy_graph(hypernymy_rels)
    else:
        hyponyms, syn2hyp = None, None

    data = args.training_data
    known_lemmas = set()
    # Path to the mapping between WordNET sense keys and synset IDs; the file must reside in the folder with the training data
    if data_source == "uniroma":
        sensekey2synset = pickle.load(open(os.path.join(data, "sensekey2synset.pkl"), "rb"))
    elif data_source == "naf" and diff_data_sources == "True":
        sensekey2synset = pickle.load(open(sensekey2synset, "rb"))
    synset2freq = {}
    # TODO fix this, for now use hardcoded path
    # else:
    #     sensekey2synset = pickle.load(open(os.path.join(data, "/home/alexander/dev/projects/BAN/neural-wsd/data/UnivRomaData/WSD_Training_Corpora/SemCor/sensekey2synset.pkl"), "rb"))
    if data_source == "naf":
        data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, pos_types, hyp2id = \
            data_ops.read_folder_semcor(data, wsd_method=wsd_method, f_lex=lexicon,
                                        hypernym_classifier=hypernym_classifier, syn2hyp=syn2hyp)
    elif data_source == "uniroma":
        hyp2id, pos_types = {}, {}
        data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq, lemma2freq = \
            data_ops.read_data_uniroma(data, sensekey2synset, wsd_method=wsd_method, f_lex=lexicon)
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
        # TODO Change this line!
        if data_source == "naf" and diff_data_sources != "True":
            val_data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, pos_types, hyp2id = \
            data_ops.read_folder_semcor(test_data, lemma2synsets, lemma2id, synset2id, mode="test",
                                        wsd_method=wsd_method, hyp2id=hyp2id, hypernym_classifier=hypernym_classifier,
                                        syn2hyp=syn2hyp)
        elif data_source == "uniroma" or diff_data_sources == "True":
            val_data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq, lemma2freq = \
            data_ops.read_data_uniroma(test_data, sensekey2synset, lemma2synsets, lemma2id, synset2id, synID_mapping,
                                       id2synset, id2pos, known_lemmas, synset2freq, wsd_method=wsd_method, mode="test",
                                       hypernym_classifier=hypernym_classifier, syn2hyp=syn2hyp, hyp2id=hyp2id)
    freq_types = {}
    if freq_classifier == "True":
        freq_types[-14] = 0
        freq_type_count = 1
        for lemma, freq in lemma2freq.iteritems():
            if freq not in freq_types:
                freq_types[freq] = freq_type_count
                freq_type_count += 1
    # get mapping from pos_ids to pos labels:
    if pos_classifier == "True":
        id2pos = {}
        for pos_label, pos_id in pos_types.iteritems():
            id2pos[pos_id] = pos_label
    # get synset embeddings if a path to a model is passed
    if sense_embeddings_src_path != "None":
        if joint_embedding == "True":
            if lemma_embeddings_src_path != None:
                sense_embeddings_model = lemma_embeddings_model
            else:
                sense_embeddings_model = word_embeddings_model
        else:
            sense_embeddings_model = KeyedVectors.load_word2vec_format(sense_embeddings_src_path, binary=False)
        sense_embeddings_full = sense_embeddings_model.syn0
        sense_embeddings = np.zeros(shape=(len(synset2id), lemma_embedding_dim), dtype=float)
        id2synset_embeddings = sense_embeddings_model.index2word
        if synset_mapping != None:
            bn2wn = pickle.load(open(synset_mapping, "rb"))
        count23 = 0
        for i, synset in enumerate(id2synset_embeddings):
            # in the first case the embeddings of the synsets use BabelNet IDs which need to be mapped to WordNet
            if synset.startswith("bn:"):
                # synset = synset[-12:]
                # bear in mind that there are 6 instances in the mapping of one BN id mapped to two WN synsets
                if synset in bn2wn:
                    synsets = bn2wn[synset]
                else:
                    continue
                for synset in synsets:
                    if synset in synset2id:
                        count23 += 1
                        sense_embeddings[synset2id[synset]] = copy(sense_embeddings_full[i])
            elif synset in synset2id:
                sense_embeddings[synset2id[synset]] = copy(sense_embeddings_full[i])
    else:
        sense_embeddings = None

    val_inputs, val_input_lemmas, val_seq_lengths, val_labels, val_words_to_disambiguate, \
    val_indices, val_lemmas_to_disambiguate, val_synsets_gold, val_pos_filters, val_pos_labels, \
    val_labels_hyp, val_indices_hyp, val_lemmas_hyp, val_pos_filters_hyp, val_freq_labels = data_ops.format_data\
                                                    (wsd_method, val_data, src2id, src2id_lemmas, synset2id,
                                                     synID_mapping, seq_width, word_embedding_case, word_embedding_input,
                                                     sense_embeddings, 0, lemma_embedding_dim, pos_types, "evaluation",
                                                     use_pos=use_pos, pos_classifier=pos_classifier,
                                                     hypernym_classifier=hypernym_classifier, hyp2id=hyp2id,
                                                     freq_classifier=freq_classifier, lemma2freq=lemma2freq, freq_types=freq_types)

    # Function to calculate the accuracy on a batch of results and gold labels
    def accuracy(logits, lemmas, synsets_gold, pos_filters, synset2id, indices=None, synID_mapping=synID_mapping,
                 pos_classifier="False", wsd_classifier="True", use_gold_pos="False", logits_pos=None, labels_pos=None,
                 hypernym_classifier="False", logits_hyp=None, labels_hyp=None, lemmas_hyp=None, pos_filters_hyp=None,
                 freq_classifier="False", logits_freq=None, labels_freq=None):

        if pos_classifier == "False":
            use_gold_pos = "True"
        accuracy_wsd = 0.0
        if wsd_classifier == "True":
            matching_cases = 0
            eval_cases = 0
            for i, logit in enumerate(logits):
                max = -10000
                max_id = -1
                gold_synsets = synsets_gold[i]
                #gold_pos = gold_synsets[0].split("-")[1]
                if pos_classifier == "True" and use_gold_pos == "False":
                    # Use fine or coarse-grained POS tagset
                    # gold_pos = pos_map[id2pos[np.argmax(logits_pos[indices[i]])]]
                    gold_pos = id2pos[np.argmax(logits_pos[indices[i]])]
                    if gold_pos in pos_map_simple:
                        gold_pos = pos_map_simple[gold_pos]
                    else:
                        gold_pos = None
                else:
                    gold_pos = pos_filters[i]
                lemma = lemmas[i]
                if lemma not in known_lemmas:
                    max_id = lemma2synsets[lemma][0]
                    # if len(lemma2synsets[lemma]) == 1:
                    #     max_id = lemma2synsets[lemma][0]
                    # elif len(lemma2synsets[lemma]) > 1:
                    #     if synset2freq[lemma] > 0:
                    #         max_id = synset2freq[lemma]
                    #     else:
                    #         max_id = random.choice(lemma2synsets[lemma])
                else:
                    for synset in lemma2synsets[lemma]:
                        id = synset2id[synset]
                        if len(synID_mapping) > 0:
                            id = synID_mapping[id]
                        # make sure we only evaluate on synsets of the correct POS category
                        if gold_pos != None and synset.split("-")[1] != gold_pos:
                            continue
                        if logit[id] > max:
                            max = logit[id]
                            max_id = synset
                #make sure there is at least one synset with a positive score
                # if max < 0:
                #     pruned_logit[max_id] = max * -1
                if max_id in gold_synsets:
                    matching_cases += 1
                eval_cases += 1
            accuracy_wsd = (100.0 * matching_cases) / eval_cases

        accuracy_pos = 0.0
        if pos_classifier == "True":
            matching_cases_pos = 0
            eval_cases_pos = 0
            for i, logit_pos in enumerate(logits_pos):
                if np.amax(labels_pos[i]) == 0:
                    continue
                if np.argmax(logit_pos) == np.argmax(labels_pos[i]):
                    matching_cases_pos += 1
                eval_cases_pos += 1
            accuracy_pos = (100.0 * matching_cases_pos) / eval_cases_pos

        accuracy_hyp = 0.0
        if hypernym_classifier == "True":
            matching_cases_hyp = 0
            eval_cases_hyp = 0
            for i, logit_hyp in enumerate(logits_hyp):
                lemma = lemmas_hyp[i]
                gold_pos = pos_filters_hyp[i]
                poss_hypernyms = []
                synsets = lemma2synsets[lemma]
                max = -10000
                max_id = -1
                for syn in synsets:
                    if syn not in syn2hyp:
                        continue
                    if syn.split("-")[1] != gold_pos:
                        continue
                    for hyp in syn2hyp[syn]:
                        if hyp in hyp2id:
                            poss_hypernyms.append(hyp2id[hyp])
                for hyp in poss_hypernyms:
                    if logit_hyp[hyp] > max:
                        max_id = hyp
                        max = logit_hyp[hyp]
                if max_id == np.argmax(labels_hyp[i]):
                    matching_cases_hyp += 1
                eval_cases_hyp += 1
            # for i, logit_hyp in enumerate(logits_hyp):
            #     if np.amax(labels_hyp[i]) == 0:
            #         continue
            #     if np.argmax(logit_hyp) == np.argmax(labels_hyp[i]):
            #         matching_cases_hyp += 1
            #     eval_cases_hyp += 1
            accuracy_hyp = (100.0 * matching_cases_hyp) / eval_cases_hyp

        accuracy_freq = 0.0
        if freq_classifier == "True":
            matching_cases_freq = 0
            eval_cases_freq = 0
            for i, logit_freq in enumerate(logits_freq):
                if np.argmax(logit_freq) == np.argmax(labels_freq[i]):
                    matching_cases_freq += 1
                eval_cases_freq += 1
            accuracy_freq = (100.0 * matching_cases_freq) / eval_cases_freq

        return accuracy_wsd, accuracy_pos, accuracy_hyp, accuracy_freq

    def accuracy_cosine_distance (logits, lemmas, synsets_gold, pos_filters):

        matching_cases = 0
        eval_cases = 0
        for i, logit in enumerate(logits):
            lemma = lemmas[i]
            poss_synsets = lemma2synsets[lemma]
            best_fit = "None"
            max_similarity = -10000.0
            # gold_pos = synsets_gold[i][0].split("-")[1]
            gold_pos = pos_filters[i]
            for j, synset in enumerate(poss_synsets):
                if synset.split("-")[1] != gold_pos:
                    continue
                syn_id = synset2id[synset]
                if syn_id >= len(sense_embeddings):
                    if max_similarity == -10000:
                        best_fit = synset
                    continue
                cos_sim = cosine_similarity(logit.reshape(1,-1), sense_embeddings[syn_id].reshape(1,-1))[0][0]
                if cos_sim > max_similarity:
                    max_similarity = cos_sim
                    best_fit = synset
            if best_fit in synsets_gold[i]:
                matching_cases += 1
            eval_cases += 1

        return (100.0 * matching_cases) / eval_cases


    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset):

        batch = data[offset:(offset+batch_size)]
        inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold, pos_filters, \
        pos_labels, hyp_labels, hyp_indices, _, _, freq_labels = data_ops.format_data(wsd_method, batch, src2id, src2id_lemmas, synset2id, synID_mapping, seq_width,
                                 word_embedding_case, word_embedding_input, sense_embeddings, dropword,
                                 lemma_embedding_dim=lemma_embedding_dim, pos_types=pos_types, use_pos=use_pos,
                                 pos_classifier=pos_classifier, hypernym_classifier=hypernym_classifier, hyp2id=hyp2id,
                                 freq_classifier=freq_classifier, lemma2freq=lemma2freq, freq_types=freq_types)
        return inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold, \
               pos_filters, pos_labels, hyp_labels, hyp_indices, freq_labels

    model = None
    if wsd_method == "similarity":
        if word_embedding_input == "wordform":
            output_embedding_dim = word_embedding_dim
        else:
            output_embedding_dim = lemma_embedding_dim
        model = ModelVectorSimilarity(word_embedding_input, output_embedding_dim, lemma_embedding_dim, vocab_size_lemmas,
                                      batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths,
                                      val_words_to_disambiguate, val_indices, val_labels, word_embedding_dim, vocab_size)
    elif wsd_method == "fullsoftmax":
        model = ModelSingleSoftmax(synset2id, word_embedding_dim, vocab_size, batch_size, seq_width, n_hidden,
                                   n_hidden_layers, val_inputs, val_input_lemmas, val_seq_lengths, val_words_to_disambiguate,
                                   val_indices, val_labels, lemma_embedding_dim, len(src2id_lemmas), wsd_classifier,
                                   pos_classifier, freq_classifier, len(pos_types), val_pos_labels, hypernym_classifier, hyp2id,
                                   val_labels_hyp, val_indices_hyp, val_freq_labels, freq_types)
    elif wsd_method == "multitask":
        if word_embedding_input == "wordform":
            output_embedding_dim = word_embedding_dim
        else:
            output_embedding_dim = lemma_embedding_dim
        model = ModelMultiTaskLearning(word_embedding_input, synID_mapping, output_embedding_dim, lemma_embedding_dim,
                                       vocab_size_lemmas, batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths,
                                       val_words_to_disambiguate, val_indices, val_labels[0], val_labels[1],
                                       word_embedding_dim, vocab_size, freq_classifier, val_freq_labels, freq_types)



    session = tf.Session()
    saver = tf.train.Saver()
    #session.run(tf.global_variables_initializer())
    if mode == "application":
        saver.restore(session, os.path.join(args.save_path, "model/model.ckpt-34400"))
        app_data = args.app_data
        data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq = \
        data_ops.read_data_uniroma(app_data, sensekey2synset, lemma2synsets, lemma2id, synset2id, synID_mapping,
                                   id2synset, id2pos, known_lemmas, synset2freq, wsd_method=wsd_method, mode="test")
        match_cases = 0
        eval_cases = 0
        for step in range(len(data) / batch_size + 1):
            offset = (step * batch_size) % (len(data))
            inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, \
            synsets_gold, pos_filters = new_batch(offset, mode="application")
            input_data = [inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices]
            fetches = run_epoch(session, model, input_data, 1, mode="application", multitask=multitask)
            acc, match_cases_count, eval_cases_count, match_be, eval_be = accuracy(fetches[0], lemmas_to_disambiguate, synsets_gold,
                                                                    pos_filters, synset2id)
            match_cases += match_cases_count
            eval_cases += eval_cases_count
        accuracy = (100.0 * match_cases) / eval_cases
        print accuracy
        exit()
    else:
        init = tf.initialize_all_variables()
        if wsd_method == "similarity":
            feed_dict = {model.place: val_labels}
            if len(word_embeddings) > 0:
                feed_dict.update({model.emb_placeholder: word_embeddings})
            if len(lemma_embeddings) > 0:
                feed_dict.update({model.emb_placeholder_lemmas: lemma_embeddings})
            session.run(init, feed_dict=feed_dict)

        elif wsd_method == "fullsoftmax":
            feed_dict={model.emb_placeholder: word_embeddings, model.place: val_labels}
            if len(lemma_embeddings) > 0:
                session.run(init, feed_dict={model.emb_placeholder: word_embeddings, model.emb_placeholder_lemmas: lemma_embeddings,
                                             model.place: val_labels})
            else:
                session.run(init, feed_dict={model.emb_placeholder: word_embeddings, model.place: val_labels})
        elif wsd_method == "multitask":
            feed_dict = {model.place_c : val_labels[0], model.place_r : val_labels[1]}
            if len(word_embeddings) > 0:
                feed_dict.update({model.emb_placeholder: word_embeddings})
            if len(lemma_embeddings) > 0:
                feed_dict.update({model.emb_placeholder_lemmas: lemma_embeddings})
            session.run(init, feed_dict=feed_dict)

    #session.run(model.set_embeddings, feed_dict={model.emb_placeholder: word_embeddings})

    print "Start of training"
    batch_loss = 0
    best_accuracy = 0.0
    if multitask == "True":
        batch_loss_r = 0
        best_accuracy_r = 0.0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    results = open(os.path.join(args.save_path, 'results.txt'), "a", 0)
    results.write(str(args) + '\n\n')
    model_path = os.path.join(args.save_path, "model")
    for step in range(training_iters):
        offset = (step * batch_size) % (len(data) - batch_size)
        inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, \
        synsets_gold, pos_filters, pos_labels, hyp_labels, hyp_indices, freq_labels = new_batch(offset)
        if (len(labels) == 0):
            continue
        input_data = [inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, pos_labels, hyp_labels,
                      hyp_indices, freq_labels]
        val_accuracy = 0.0
        if (step % 100 == 0):
            print "Step number " + str(step)
            fetches = run_epoch(session, model, input_data, keep_prob, mode="val", multitask=multitask)
            if (fetches[1] is not None):
                batch_loss += fetches[1]
            if multitask == "True" and fetches[2] is not None:
                batch_loss_r += fetches[2]
            results.write('EPOCH: %d' % step + '\n')
            results.write('Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss/100.0) + '\n')
            if multitask == "True":
                results.write('Averaged minibatch loss (similarity) at step ' + str(step) + ': ' + str(batch_loss_r / 100.0) + '\n')
            if wsd_method == "similarity":
                val_accuracy = accuracy_cosine_distance(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold, val_pos_filters)
                results.write('Minibatch accuracy: ' + str(accuracy_cosine_distance(fetches[2], lemmas_to_disambiguate,
                                                                                    synsets_gold, pos_filters)) + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
                # Uncomment lines below in order to save the array with the modified word embeddings
                # if val_accuracy > 55.0 and val_accuracy > best_accuracy:
                #     with open(os.path.join(args.save_path, 'embeddings.pkl'), 'wb') as output:
                #                 pickle.dump(fetches[-1], output, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(args.save_path, 'src2id_lemmas.pkl'), 'wb') as output:
                #         pickle.dump(src2id_lemmas, output, pickle.HIGHEST_PROTOCOL)
            elif wsd_method == "fullsoftmax":
                val_accuracy, val_accuracy_pos, val_accuracy_hyp, val_accuracy_freq = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters, synset2id, val_indices, pos_classifier=pos_classifier,
                                                          wsd_classifier=wsd_classifier, logits_pos=fetches[5],
                                                          labels_pos=val_pos_labels, hypernym_classifier=hypernym_classifier,
                                                          logits_hyp=fetches[7], labels_hyp=val_labels_hyp, lemmas_hyp=val_lemmas_hyp,
                                                          pos_filters_hyp=val_pos_filters_hyp)
                results.write('Minibatch accuracy: ' + str(accuracy(fetches[2], lemmas_to_disambiguate, synsets_gold,
                                                                    pos_filters, synset2id, indices, pos_classifier=pos_classifier,
                                                                    wsd_classifier=wsd_classifier, logits_pos=fetches[4],
                                                                    labels_pos=pos_labels)[0])
                              + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
            elif wsd_method == "multitask":
                val_accuracy, _, _, val_accuracy_freq = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold, val_pos_filters,
                                        synset2id, synID_mapping, freq_classifier=freq_classifier, logits_freq=fetches[8], labels_freq=val_freq_labels)
                results.write('Minibatch classification accuracy: ' +
                              str(accuracy(fetches[3], lemmas_to_disambiguate, synsets_gold, pos_filters,
                                           synset2id, synID_mapping)[0]) + '\n')
                results.write('Validation classification accuracy: ' + str(val_accuracy) + '\n')
                val_accuracy_r = accuracy_cosine_distance(fetches[6], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters)
                results.write('Minibatch regression accuracy: ' +
                              str(accuracy_cosine_distance(fetches[5], lemmas_to_disambiguate, synsets_gold,
                                                           pos_filters)) + '\n')
                results.write('Validation regression accuracy: ' + str(val_accuracy_r) + '\n')

                # ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.val_logits,
                #        model.output_emb, model.val_output_emb]
            print "Validation accuracy: " + str(val_accuracy)
            if freq_classifier == "True":
                print "Validation accuracy for frequency classification is: " + str(val_accuracy_freq)
            if pos_classifier == "True":
                print "Validation accuracy for POS: " + str(val_accuracy_pos)
                results.write('Validation accuracy for POS: ' + str(val_accuracy_pos) + '\n')
            if hypernym_classifier == "True":
                print "Validation accuracy for hypernyms: " + str(val_accuracy_hyp)
                results.write('Validation accuracy for hypernyms: ' + str(val_accuracy_hyp) + '\n')
            batch_loss = 0.0
            if wsd_method == "multitask":
                batch_loss_r = 0.0
        else:
            fetches = run_epoch(session, model, input_data, keep_prob, mode="train", multitask=multitask)
            if (fetches[1] is not None):
                batch_loss += fetches[1]
            if multitask == "True" and fetches[2] is not None:
                batch_loss_r += fetches[2]

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if pos_classifier == "True" and wsd_classifier == "True":
                val_accuracy_gold_pos, _, _, _ = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters, synset2id, val_indices,
                                                          pos_classifier=pos_classifier, use_gold_pos="True",
                                                          logits_pos=fetches[5], labels_pos=val_pos_labels)
                results.write('Validation classification accuracy with gold POS: ' + str(val_accuracy_gold_pos) + '\n')

        if multitask == "True" and val_accuracy_r > best_accuracy_r:
            best_accurary_r = val_accuracy_r


        if (args.save_path != "None" and step == 25000 or step > 25000 and val_accuracy == best_accuracy):
            for file in os.listdir(model_path):
                os.remove(os.path.join(model_path, file))
            saver.save(session, os.path.join(args.save_path, "model/model.ckpt"), global_step=step)
            if (step == 25000):
                with open(os.path.join(args.save_path, 'lemma2synsets.pkl'), 'wb') as output:
                    pickle.dump(lemma2synsets, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'lemma2id.pkl'), 'wb') as output:
                    pickle.dump(lemma2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'synset2id.pkl'), 'wb') as output:
                    pickle.dump(synset2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2synset.pkl'), 'wb') as output:
                    pickle.dump(id2synset, output, pickle.HIGHEST_PROTOCOL)

    results.write('\n\n\n' + 'Best result is: ' + str(best_accuracy))
    if multitask == "True":
        results.write('\n\n\n' + 'Best result (similarity) is: ' + str(best_accuracy_r))
    results.close()
