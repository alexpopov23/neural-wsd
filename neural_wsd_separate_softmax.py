import argparse
import random

import tensorflow as tf
import numpy as np

import word2vec_optimized_utf8 as w2v
import data_ops

from functools import partial
from sys import getsizeof

class Model:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, is_first, lemma2synsets, lemma2id, src2id, word_embeddings, word_embedding_dim, batch_size, seq_width, n_hidden,
                 val_size, val_inputs, val_seq_lengths, val_flags, val_labels):
        W = tf.get_variable(name="W", shape=[len(src2id), word_embedding_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(word_embeddings), trainable=True)
        self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_model_flags = tf.placeholder(tf.bool, shape=[batch_size, seq_width, len(lemma2synsets)])
        self.train_labels = [[tf.placeholder(tf.int32, shape=[None]) for _ in range(seq_width)] for _ in range(batch_size)]
        # self.val_inputs = tf.constant(val_inputs, tf.int32)
        # self.val_seq_lengths = tf.constant(val_seq_lengths, tf.int32)
        # self.val_flags = tf.constant(val_flags, tf.bool)
        # self.val_labels = [[tf.constant(label, dtype=tf.int32, shape=[len(label)]) for label in sent] for sent in val_labels]
        # self.val_inputs = tf.placeholder(tf.int32, shape=[val_size, seq_width])
        # self.val_seq_lengths = tf.placeholder(tf.int32, shape=[val_size])
        # self.val_flags = tf.placeholder(tf.bool, shape=[val_size, seq_width, len(lemma2synsets)])
        # self.val_labels = [[tf.placeholder(tf.int32, shape=[None]) for label in sent] for sent in val_labels]

        for i in range(len(self.train_labels)):
            for j in range(len(self.train_labels[i])):
                print self.train_labels[i][j]

        reuse = None if is_first else True

        # create parameters for all word sense models
        with tf.variable_scope("word-sense-models") as scope:
            # Bidirectional recurrent neural network with LSTM cells
            initializer = tf.random_uniform_initializer(-1, 1)
            #with tf.variable_scope('forward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
            fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
            #with tf.variable_scope('backward'):
                # TODO: Use state_is_tuple=True
                # TODO: add dropout
            bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)

            for lemma, synsets in lemma2synsets.iteritems():
                # with tf.variable_scope(lemma, True):
                if len(synsets) > 1:
                    weights = tf.get_variable(name=(lemma.replace("\'", "_") + "-w"),
                                              shape=[2 * n_hidden, len(synsets)], dtype=tf.float32)
                    biases = tf.get_variable(name=(lemma.replace("\'", "_") + "-b"), shape=[len(synsets)],
                                             dtype=tf.float32)
            #scope.reuse_variables()

            def biRNN_WSD (inputs, seq_lengths, flags, labels, train):

                embedded_inputs = tf.nn.embedding_lookup(W, inputs)
                embedded_inputs = tf.unpack(tf.transpose(embedded_inputs, [1, 0, 2]))

                # Get the blstm cell output
                outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, embedded_inputs, dtype="float32",
                                                        sequence_length=seq_lengths)
                outputs = tf.transpose(outputs, [1, 0, 2])

                scope.reuse_variables()

                def calc_cost_for_word_sense (rnn_output, weights, biases, label, input, flag, debug):
                    debug.append((rnn_output,weights,biases,input, flag, label, weights.name))
                    logit = tf.matmul(tf.reshape(rnn_output, [1, 2*n_hidden]), weights) + biases
                    logit = tf.reshape(logit, [-1])
                    # print "Logit is:"
                    # print logit
                    # print "Label is:"
                    # print label
                    #cost = tf.nn.softmax_cross_entropy_with_logits(logit, label)
                    def f1 (_logit, _label):
                        result = tf.nn.softmax_cross_entropy_with_logits(_logit, _label)
                        return result
                    def f2 ():
                        return tf.zeros([1])
                    pred = tf.equal(tf.gather_nd(tf.shape(biases), [0]), tf.gather_nd(tf.shape(label), [0]))
                    cost = tf.cond(pred, partial(f1, _logit=logit, _label=label), f2)

                    return cost, logit

                costs = []
                logits = []
                debug = []
                for i, sentence in enumerate(tf.unpack(outputs)):
                    curr_costs = []
                    curr_logits = []
                    for j, word in enumerate(tf.unpack(sentence)):
                        pred_fn_pairs = []
                        print "At word position" + str(j) + "in sentence " + str(i)
                        input = tf.gather_nd(inputs, [i, j])
                        for lemma in lemma2synsets.iterkeys():
                            if len(lemma2synsets[lemma]) > 1:
                                #with tf.variable_scope(lemma, True):
                                pred_fn_pairs.append((tf.gather_nd(flags, [i, j, lemma2id[lemma]]),
                                                      partial(calc_cost_for_word_sense,
                                                              rnn_output=word,
                                                              weights=tf.get_variable(name=(lemma.replace("\'", "_")+"-w")),
                                                              biases=tf.get_variable(name=(lemma.replace("\'", "_") + "-b")),
                                                              label=labels[i][j],
                                                              input=input,
                                                              flag=tf.gather_nd(flags, [i, j, lemma2id[lemma]]),
                                                              debug=debug
                                                              )))
                        default_call = lambda: (tf.zeros(shape=[],dtype=tf.float32), tf.zeros([1], dtype=tf.float32))
                        word_cost, logit = tf.case(pred_fn_pairs=pred_fn_pairs, default=default_call)
                        #TODO remove zero costs
                        curr_costs.append(word_cost)
                        curr_logits.append(logit)
                    costs.append(curr_costs)
                    logits.append(curr_logits)
                costs = tf.pack(costs)
                #self.predictions = tf.nn.softmax(self.logits)
                #self.reduced_cost = tf.reduce_mean(tf.reduce_sum(self.costs))
                reduced_cost = tf.reduce_mean(costs)
                #self.gradients = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(self.reduced_cost)
                #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(reduced_cost)

                # if (train == True):
                #     return reduced_cost, logits
                # else:
                #     return logits

                return reduced_cost, logits

            self.loss, self.logits = biRNN_WSD(self.train_inputs, self.train_seq_lengths,
                                                           self.train_model_flags, self.train_labels, True)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            # scope.reuse_variables()
            # self.val_logits = biRNN_WSD(self.val_inputs, self.val_seq_lengths, self.val_flags, self.val_labels, False)

def run_epoch(session, model, data, mode):

    inputs = data[0]
    seq_lengths = data[1]
    labels = data[2]
    words_to_disambiguate = data[3]
    feed_dict = { model.train_inputs : inputs, model.train_seq_lengths : seq_lengths,
                  model.train_model_flags : words_to_disambiguate }
    for tf_batch_labels, batch_labels in zip(model.train_labels, labels):
        for tf_label, label in zip(tf_batch_labels, batch_labels):
            feed_dict.update({tf_label : label})
    if mode == "train":
        #ops = [model.reduced_cost, model.logits, model.train_op, model.costs, model.logits]
        ops = [model.train_op, model.loss, model.logits]
    elif mode == "val":
        ops = [model.logits]
    #fetches = []
    #try:
    fetches = session.run(ops, feed_dict=feed_dict)
    #except:
    #print "Could not feed tensors in dictionary"

    return fetches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural POS tagger.')
    parser.add_argument('-word_embeddings_src_model', dest='word_embeddings_src_save_path', required=False,
                        help='The path to the pretrained model with the word embeddings (for the source language).')
    parser.add_argument('-word_embeddings_src_train_data', dest='word_embeddings_src_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the source language.')
    parser.add_argument('-word_embedding_dim', dest='word_embedding_dim', required=False,
                        help='Size of the word embedding vectors.')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.3,
                        help='How fast should the network learn.')
    parser.add_argument('-training_iterations', dest='training_iters', required=False, default=10000,
                        help='How many iterations should the network train for.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=100,
                        help='Size of the hidden layer.')
    parser.add_argument('-sequence_width', dest='seq_width', required=False, default=50,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-training_data', dest='training_data', required=True, default="brown",
                        help='The path to the gold corpus used for training/testing.')
    parser.add_argument('-lexicon', dest='lexicon', required=True, default="None",
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-save_path', dest='save_path', required=False, default="None",
                        help='Path to where the model should be saved.')

    # read the parameters for the model and the data
    args = parser.parse_args()
    word_embedding_dim = int(args.word_embedding_dim)
    word2id = {} # map word strings to ids
    word_embeddings = {} # store the normalized embeddings; keys are integers (0 to n)
    #TODO load the vectors from a saved structure, this TF graph below is pointless
    with tf.Graph().as_default(), tf.Session() as session:
        opts = w2v.Options()
        opts.train_data = args.word_embeddings_src_train_data
        opts.save_path = args.word_embeddings_src_save_path
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

    # Network Parameters
    learning_rate = float(args.learning_rate) # Update rate for the weights
    training_iters = int(args.training_iters) # Number of training steps
    batch_size = int(args.batch_size) # Number of sentences passed to the network in one batch
    seq_width = int(args.seq_width) # Max sentence length (longer sentences are cut to this length)
    n_hidden = int(args.n_hidden) # Number of features/neurons in the hidden layer
    embedding_size = word_embedding_dim
    vocab_size = len(word2id)
    lexicon = args.lexicon

    data = args.training_data
    data, lemma2synsets, lemma2id = data_ops.read_folder_semcor(data)
    random.shuffle(data)
    partition = int(len(data) * 0.9)
    train_data = data[:partition]
    val_data = data[partition:]
    val_inputs, val_seq_lengths, val_labels, val_words_to_disambiguate = \
        data_ops.format_data(val_data, src2id, lemma2synsets, lemma2id, seq_width)
    val_data = [val_inputs, val_seq_lengths, val_labels, val_words_to_disambiguate]
    #test_data = data[2]

    # Function to calculate the accuracy on a batch of results and gold labels
    def accuracy(logits, labels, seq_lengths, words_to_disambiguate):

        matching_cases = 0
        eval_cases = 0
        for i, sentence in enumerate(words_to_disambiguate):
            for j, flag in enumerate(sentence):
                if j >= seq_lengths[i]:
                    break
                if np.amax(flag) == True and len(labels[i][j]) > 1:
                    # TODO FIX the [0]-bit - makes no sense, figure out how to pass it neatly
                    if np.argmax(logits[i][0][j]) == np.argmax(labels[i][j]):
                        matching_cases += 1
                    eval_cases += 1

        return (100.0 * matching_cases) / eval_cases

    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset):

        batch = data[offset:(offset+batch_size)]
        inputs, seq_lengths, labels, words_to_disambiguate = data_ops.format_data(batch, src2id, lemma2synsets, lemma2id, seq_width)
        return inputs, seq_lengths, labels, words_to_disambiguate

    model = Model(True, lemma2synsets, lemma2id, src2id, word_embeddings, word_embedding_dim, batch_size, seq_width, n_hidden,
                  len(val_inputs), val_inputs, val_seq_lengths, val_words_to_disambiguate, val_labels)
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    batch_loss = 0
    for step in range(training_iters):
        offset = (step * batch_size) % (len(data) - batch_size)
        inputs, seq_lengths, labels, words_to_disambiguate = new_batch(offset)
        input_data = [inputs, seq_lengths, labels, words_to_disambiguate]
        train_fetches = run_epoch(session, model, input_data, mode="train")

        if (train_fetches[1] is not None):
            batch_loss += train_fetches[1]
        #print train_fetches

        if (step % 100) == 0:
            print 'EPOCH: %d' % step
            print 'Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss/100.0)
            batch_loss = 0.0

        if (step % 500 == 0):
            val_logits = []
            for i in xrange(len(val_inputs)):
                val_fetches = run_epoch(session, model, [[val_inputs[i]], [val_seq_lengths[i]], [val_labels[i]],
                                                         [val_words_to_disambiguate[i]]], mode="val")
                val_logits.extend(val_fetches)
            #print 'Minibatch loss at step ' + str(step) + ': ' + str(train_fetches[0])
            #print 'Minibatch accuracy: ' + str(accuracy(train_fetches[1], labels, seq_lengths, words_to_disambiguate))
            print 'Validation accuracy: ' + str(accuracy(
                val_logits, val_labels, val_seq_lengths, val_words_to_disambiguate))