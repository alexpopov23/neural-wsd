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


class ModelSingleSoftmax:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, synset2id, word_embedding_dim, vocab_size,
                 batch_size, seq_width, n_hidden, n_hidden_layers,
                 val_inputs, val_input_lemmas, val_seq_lengths, val_flags, val_indices, val_labels,
                 lemma_embedding_dim, vocab_size_lemmas):
        self.emb_placeholder = tf.placeholder(tf.float32, shape=[vocab_size, word_embedding_dim])
        self.embeddings = tf.Variable(self.emb_placeholder)
        self.set_embeddings = tf.assign(self.embeddings, self.emb_placeholder, validate_shape=False)
        if vocab_size_lemmas > 0:
            self.emb_placeholder_lemmas = tf.placeholder(tf.float32, shape=[vocab_size_lemmas, lemma_embedding_dim])
            self.embeddings_lemmas = tf.Variable(self.emb_placeholder_lemmas)
            self.set_embeddings_lemmas = tf.assign(self.embeddings_lemmas, self.emb_placeholder_lemmas, validate_shape=False)
        #TODO pick an initializer
        self.weights = tf.get_variable(name="softmax-w", shape=[2*n_hidden, len(synset2id)], dtype=tf.float32)
        self.biases = tf.get_variable(name="softmax-b", shape=[len(synset2id)], dtype=tf.float32)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_inputs_lemmas = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width])
        self.train_labels = tf.placeholder(tf.int32, shape=[None, len(synset2id)])
        self.train_indices = tf.placeholder(tf.int32, shape=[None])
        self.val_inputs = tf.constant(val_inputs, tf.int32)
        if vocab_size_lemmas > 0:
            self.val_inputs_lemmas = tf.constant(val_input_lemmas, tf.int32)
        self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32)
        self.val_flags = tf.constant(val_flags, tf.bool)
        self.place = tf.placeholder(tf.int32, shape=val_labels.shape)
        self.val_labels = tf.Variable(self.place)
        self.val_indices = tf.constant(val_indices, tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        def embed_inputs (input_words, input_lemmas=None):

            embedded_inputs = tf.nn.embedding_lookup(self.embeddings, input_words)
            if input_lemmas != None:
                embedded_inputs_lemmas = tf.nn.embedding_lookup(self.embeddings_lemmas, input_lemmas)
                embedded_inputs = tf.concat([embedded_inputs, embedded_inputs_lemmas], 2)

            return embedded_inputs

        def biRNN_WSD (embedded_inputs, seq_lengths, indices, weights, biases, labels, is_training, keep_prob):

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
                target_outputs = tf.gather(rnn_outputs, indices)
                logits = tf.matmul(target_outputs, weights) + biases
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cost = tf.reduce_mean(losses)

            return cost, logits, losses

        # if lemma embeddings are passed, then concatenate them with the word embeddings
        if vocab_size_lemmas > 0:
            embedded_inputs = embed_inputs(self.train_inputs, self.train_inputs_lemmas)
        else:
            embedded_inputs = embed_inputs(self.train_inputs)
        self.cost, self.logits, self.losses = biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                                           self.weights, self.biases, self.train_labels, True, self.keep_prob)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        if vocab_size_lemmas > 0:
            embedded_inputs = embed_inputs(self.val_inputs, self.val_inputs_lemmas)
        else:
            embedded_inputs = embed_inputs(self.val_inputs)
        tf.get_variable_scope().reuse_variables()
        _, self.val_logits, _ = biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                                       self.weights, self.biases, self.val_labels, False, 1.0)

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
                 val_labels_classification, val_labels_regression, word_embedding_dim, vocab_size_wordforms):

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
                       labels_c, labels_r, is_training, keep_prob=1.0):

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
                cost = cost_c + cost_r

            return cost, cost_c, cost_r, output_c, output_r


        # if lemma embeddings are passed, then concatenate them with the word embeddings
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas, self.train_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.train_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.train_inputs)
        self.cost, self.cost_c, self.cost_r, self.logits, self.output_emb = \
            biRNN_WSD(embedded_inputs, self.train_seq_lengths, self.train_indices,
                      self.weights_classification, self.biases_classification,
                      self.weights_regression, self.biases_regression,
                      self.train_labels_classification, self.train_labels_regression,
                      True, self.keep_prob)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        if input_mode == "joint":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas, self.val_inputs)
        elif input_mode == "lemma":
            embedded_inputs = embed_inputs(self.val_inputs_lemmas)
        elif input_mode == "wordform":
            embedded_inputs = embed_inputs(self.val_inputs)
        tf.get_variable_scope().reuse_variables()
        _, _, _, self.val_logits, self.val_output_emb = \
            biRNN_WSD(embedded_inputs, self.val_seq_lengths, self.val_indices,
                      self.weights_classification, self.biases_classification,
                      self.weights_regression, self.biases_regression,
                      self.val_labels_classification, self.val_labels_regression,
                      False)

def run_epoch(session, model, data, keep_prob, mode, multitask="False"):

    feed_dict = {}
    if mode != "application":
        inputs = data[0]
        input_lemmas = data[1]
        seq_lengths = data[2]
        labels = data[3]
        words_to_disambiguate = data[4]
        indices = data[5]
        feed_dict = { model.train_seq_lengths : seq_lengths,
                      model.train_model_flags : words_to_disambiguate,
                      model.train_indices : indices,
                      model.keep_prob : keep_prob}
        if multitask == "True":
            feed_dict.update({model.train_labels_classification: labels[0]})
            feed_dict.update({model.train_labels_regression: labels[1]})
        else:
            feed_dict.update({model.train_labels: labels})
        if len(inputs) > 0:
            feed_dict.update({model.train_inputs: inputs})
        if len(input_lemmas) > 0:
            feed_dict.update({model.train_inputs_lemmas : input_lemmas})
    if mode == "train":
        if multitask == "True":
            ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.output_emb]
        else:
            ops = [model.train_op, model.cost, model.logits]
    elif mode == "val":
        if multitask == "True":
            ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.val_logits,
                   model.output_emb, model.val_output_emb]
        else:
            ops = [model.train_op, model.cost, model.logits, model.val_logits, model.embeddings_lemmas]
    elif mode == "application":
        ops = [model.val_logits]
    fetches = session.run(ops, feed_dict=feed_dict)

    return fetches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural WSD tagger.')
    parser.add_argument("-data_source", dest="data_source", required=False, default="uniroma",
                        help="Which corpus are we using? Needed to determine how to read the data. Options: naf, uniroma")
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training run or an application run? Options: train, application")
    parser.add_argument('-wsd_method', dest='wsd_method', required=True, default="fullsoftmax",
                        help='Which method is used for the final, WSD step: similarity or fullsoftmax?')
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
    n_hidden_layers = int(args.n_hidden_layers) # Number of features/neurons in the hidden layer
    embedding_size = word_embedding_dim
    vocab_size = len(src2id)
    vocab_size_lemmas = len(src2id_lemmas)
    lexicon_mode = args.lexicon_mode
    lexicon = args.lexicon
    partition_point = float(args.partition_point)
    keep_prob = float(args.keep_prob)
    dropword = float(args.dropword)

    data = args.training_data
    known_lemmas = set()
    # Path to the mapping between WordNET sense keys and synset IDs; the file must reside in the folder with the training data
    sensekey2synset = pickle.load(open(os.path.join(data, "sensekey2synset.pkl"), "rb"))
    if data_source == "naf":
        data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos = \
            data_ops.read_folder_semcor(data, lexicon_mode=lexicon_mode, f_lex=lexicon)
    elif data_source == "uniroma":
        data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq = \
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
        if data_source == "naf":
            val_data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos = \
            data_ops.read_folder_semcor(test_data, lemma2synsets, lemma2id, synset2id, mode="test")
        elif data_source == "uniroma":
            val_data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq = \
            data_ops.read_data_uniroma(test_data, sensekey2synset, lemma2synsets, lemma2id, synset2id, synID_mapping,
                                       id2synset, id2pos, known_lemmas, synset2freq, wsd_method=wsd_method, mode="test")
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
        sense_embeddings = np.zeros(shape=(len(synset2id), 300), dtype=float)
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
    val_indices, val_lemmas_to_disambiguate, val_synsets_gold = data_ops.format_data\
                                                    (wsd_method, val_data, src2id, src2id_lemmas, synset2id,
                                                     synID_mapping, seq_width, word_embedding_case, word_embedding_input,
                                                     sense_embeddings, 0, "evaluation")

    # Function to calculate the accuracy on a batch of results and gold labels
    def accuracy(logits, lemmas, synsets_gold, synset2id, synID_mapping=synID_mapping):

        matching_cases = 0
        eval_cases = 0
        for i, logit in enumerate(logits):
            max = -10000
            max_id = -1
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
                    if len(synID_mapping) > 0:
                        id = synID_mapping[id]
                    # make sure we only evaluate on synsets of the correct POS category
                    if synset.split("-")[1] != gold_pos:
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

        return (100.0 * matching_cases) / eval_cases

    def accuracy_cosine_distance (logits, lemmas, synsets_gold):

        matching_cases = 0
        eval_cases = 0
        for i, logit in enumerate(logits):
            lemma = lemmas[i]
            poss_synsets = lemma2synsets[lemma]
            best_fit = "None"
            max_similarity = -10000.0
            gold_pos = synsets_gold[i][0].split("-")[1]
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
        inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold = \
            data_ops.format_data(wsd_method, batch, src2id, src2id_lemmas, synset2id, synID_mapping, seq_width,
                                 word_embedding_case, word_embedding_input, sense_embeddings, dropword)
        return inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas, synsets_gold

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
                                   val_indices, val_labels, lemma_embedding_dim, len(src2id_lemmas))
    elif wsd_method == "multitask":
        if word_embedding_input == "wordform":
            output_embedding_dim = word_embedding_dim
        else:
            output_embedding_dim = lemma_embedding_dim
        model = ModelMultiTaskLearning(word_embedding_input, synID_mapping, output_embedding_dim, lemma_embedding_dim,
                                       vocab_size_lemmas, batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths,
                                       val_words_to_disambiguate, val_indices, val_labels[0], val_labels[1],
                                       word_embedding_dim, vocab_size)



    session = tf.Session()
    saver = tf.train.Saver()
    #session.run(tf.global_variables_initializer())
    if mode == "application":
        saver.restore(session, os.path.join(args.save_path, "model.ckpt"))
        #TODO: finish this module
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
        if wsd_method == "similarity":
            feed_dict = {model.place: val_labels}
            if len(word_embeddings) > 0:
                feed_dict.update({model.emb_placeholder: word_embeddings})
            if len(lemma_embeddings) > 0:
                feed_dict.update({model.emb_placeholder_lemmas: lemma_embeddings})
            session.run(init, feed_dict=feed_dict)

        elif wsd_method == "fullsoftmax":
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
    for step in range(training_iters):
        offset = (step * batch_size) % (len(data) - batch_size)
        inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, synsets_gold = new_batch(offset)
        if (len(labels) == 0):
            continue
        input_data = [inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices]
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
                val_accuracy = accuracy_cosine_distance(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold)
                results.write('Minibatch accuracy: ' + str(accuracy_cosine_distance(fetches[2], lemmas_to_disambiguate, synsets_gold)) + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
                # Uncomment lines below in order to save the array with the modified word embeddings
                # if val_accuracy > 55.0 and val_accuracy > best_accuracy:
                #     with open(os.path.join(args.save_path, 'embeddings.pkl'), 'wb') as output:
                #                 pickle.dump(fetches[-1], output, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(args.save_path, 'src2id_lemmas.pkl'), 'wb') as output:
                #         pickle.dump(src2id_lemmas, output, pickle.HIGHEST_PROTOCOL)
            elif wsd_method == "fullsoftmax":
                val_accuracy = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold, synset2id)
                results.write('Minibatch accuracy: ' + str(accuracy(fetches[2], lemmas_to_disambiguate, synsets_gold, synset2id))
                              + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
            elif wsd_method == "multitask":
                val_accuracy = accuracy(fetches[4], val_lemmas_to_disambiguate, val_synsets_gold, synset2id, synID_mapping)
                results.write('Minibatch classification accuracy: ' +
                              str(accuracy(fetches[3], lemmas_to_disambiguate, synsets_gold, synset2id, synID_mapping)) + '\n')
                results.write('Validation classification accuracy: ' + str(val_accuracy) + '\n')
                val_accuracy_r = accuracy_cosine_distance(fetches[6], val_lemmas_to_disambiguate, val_synsets_gold)
                results.write('Minibatch regression accuracy: ' +
                              str(accuracy_cosine_distance(fetches[5], lemmas_to_disambiguate, synsets_gold)) + '\n')
                results.write('Validation regression accuracy: ' + str(val_accuracy_r) + '\n')

                # ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.val_logits,
                #        model.output_emb, model.val_output_emb]
            print "Validation accuracy: " + str(val_accuracy)
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
        if multitask == "True" and val_accuracy_r > best_accuracy_r:
            best_accurary_r = val_accuracy_r

        # if (args.save_path != "None" and step == 25000 or step > 25000 and val_accuracy == best_accuracy):
        #     saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=step)
        #     if (step == 25000):
        #         with open(os.path.join(args.save_path, 'lemma2synsets.pkl'), 'wb') as output:
        #             pickle.dump(lemma2synsets, output, pickle.HIGHEST_PROTOCOL)
        #         with open(os.path.join(args.save_path, 'lemma2id.pkl'), 'wb') as output:
        #             pickle.dump(lemma2id, output, pickle.HIGHEST_PROTOCOL)
        #         with open(os.path.join(args.save_path, 'synset2id.pkl'), 'wb') as output:
        #             pickle.dump(synset2id, output, pickle.HIGHEST_PROTOCOL)
        #         with open(os.path.join(args.save_path, 'id2synset.pkl'), 'wb') as output:
        #             pickle.dump(id2synset, output, pickle.HIGHEST_PROTOCOL)

    results.write('\n\n\n' + 'Best result is: ' + str(best_accuracy))
    if multitask == "True":
        results.write('\n\n\n' + 'Best result (similarity) is: ' + str(best_accuracy_r))
    results.close()
