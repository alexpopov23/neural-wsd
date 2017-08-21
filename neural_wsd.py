import argparse
import random

import tensorflow as tf
import numpy as np

import word2vec_optimized_utf8 as w2v
import data_ops


class WSDNet:
    def __init__(self, input, target_id, n_senses, n_hidden):
        #self.input = tf.placeholder(tf.float32, shape=[1, 2*n_hidden])
        self.input = input
        self.label = tf.placeholder(tf.int32, shape=[n_senses])
        #TODO reseach more whether the reuse option should be used
        #target_scope_reuse = None if is_training else True
        with tf.variable_scope(str(target_id), True):
            weights = tf.get_variable(name=target_id+"-w", shape=[2*n_hidden, n_senses], dtype=tf.float32)
            biases = tf.get_variable(name=target_id+"-b", shape=[n_senses], dtype=tf.float32)
            self.logit = tf.matmul(self.input, weights) + biases
            self.cost = tf.nn.softmax_cross_entropy_with_logits(self.logit, self.label)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

class BLSTMNet:
    #TODO make model work with batches (no reason not to use them before the WSD part, I think)
    def __init__(self, is_first, src2id, word_embeddings, word_embedding_dim, batch_size, seq_width, n_hidden):
        W = tf.get_variable(name="W", shape=[len(src2id), word_embedding_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(word_embeddings), trainable=True)
        self.inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_width])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])

        embedded_inputs = tf.nn.embedding_lookup(W, self.inputs)
        embedded_inputs = tf.unpack(tf.transpose(embedded_inputs, [1, 0, 2]))
        reuse = None if is_first else True

        # Bidirectional recurrent neural network with LSTM cells
        #def BiRNN(inputs, _seq_length):
        #inputs = tf.unpack(self.inputs)
        initializer = tf.random_uniform_initializer(-1, 1)
        with tf.variable_scope('forward', reuse):
            # TODO: Use state_is_tuple=True
            # TODO: add dropout
            fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
        with tf.variable_scope('backward', reuse):
            # TODO: Use state_is_tuple=True
            # TODO: add dropout
            bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
        # Get the blstm cell output
        self.outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, embedded_inputs, dtype="float32",
                                                sequence_length=self.seq_lengths)
        self.outputs = tf.transpose(self.outputs, [1, 0, 2])

class Trainer:
    def __init__(self):
        self.costs = [tf.placeholder(tf.float32)]
        reduced_cost = tf.reduce_mean(self.costs)
        #self.accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(self.sense_ids, 1)), tf.float32))
        #TODO capped gradients?
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(reduced_cost)

def run_epoch(session, blstm, models, trainer, data, mode):
    #TODO blstm and trainer models should be initialized outside of this function
    #blstm = BLSTMNet(True, src2id, word_embeddings, word_embedding_dim, seq_width, n_hidden)
    #for i, data_point in enumerate(data):
    inputs = data[0]
    seq_lengths = data[1]
    labels = data[2]
    words_to_disambiguate = data[3]
    feed_dict = { blstm.inputs : inputs, blstm.seq_lengths : seq_lengths }
    outputs = session.run(blstm.outputs, feed_dict=feed_dict)

    costs = []
    for i, output in enumerate(outputs):
        logits = []
        # disambiguate each word separately:
        for j, word_position in words_to_disambiguate[i]:
            wsdnet = models[i][j]
            feed_dict_sense = { wsdnet.input : output[word_position],
                                wsdnet.label : labels[i][word_position]}
            _, cost, logit = session.run([wsdnet.train_op, wsdnet.cost, wsdnet.logit], feed_dict=feed_dict_sense)
            #session.partial_run()
            costs.append(cost)
            logits.append(logit)
    #feed_dict = { tf_cost : cost for tf_cost, cost in zip(trainer.costs, costs)}
    #session.run(trainer.train_op, feed_dict=feed_dict)
    #trainer = Trainer()

    #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(self.sense_ids, 1)), tf.float32))

    '''
    with tf.variable_scope("BiRNN") as scope:
        logits = BiRNN(train_embeddings, tf_train_seq_length)
        losses = []
        logits = tf.unpack(logits)

        # for i, batch_results in enumerate(logits):
        #    batch_results = tf.unpack(batch_results)
        #    for j, step_result in enumerate(batch_results):
        #        label = tf.gather_nd(tf_train_labels[i], [j])
        #        loss = tf.nn.softmax_cross_entropy_with_logits([step_result], [label])
        #        losses.append(loss)
        def calc_loss(transformed_value, (result, label)):
            final_value = tf.nn.softmax_cross_entropy_with_logits([result], [label])
            return final_value

        for i, batch_results in enumerate(logits):
            # labels = tf_train_labels[i]
            labels = tf.gather(tf_train_labels[i], tf_training_points[i])
            initializer = tf.placeholder(dtype=tf.float32, shape=[None, None])
            loss = tf.scan(calc_loss, [batch_results, labels], initializer=initializer)
            losses.append(loss)
        loss = tf.reduce_mean(losses)

        # calculate gradients, clip them and update model in separate steps
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad != None]
        optimizer_t = optimizer.apply_gradients(capped_gradients)
        train_prediction = tf.nn.softmax(logits)
    '''

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
    #if word_embedding_dim == 0:
    #   print "No embedding model given as parameter!"
    #   exit(1)

    data = args.training_data
    data, lemma2synsets = data_ops.read_folder_semcor(data)
    random.shuffle(data)

    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset):

        batch = data[offset:(offset+batch_size)]
        batch_input = data_ops.format_data(batch, src2id, lemma2synsets, seq_width)
        return batch_input

    # Function to calculate the accuracy on a batch of results and gold labels '''
    def accuracy (predictions, labels, seq_lengths):

        reshaped_labels = np.reshape(np.transpose(labels, (1, 0, 2)), (-1, target_vocab_size))
        matching_cases = 0
        eval_cases = 0
        # Do not count results beyond the end of a sentence (in the case of sentences shorter than 50 words)
        for i in xrange(seq_width):
            for j in xrange(batch_size):
                if i+1 > seq_lengths[j]:
                    continue
                if np.argmax(reshaped_labels[i*batch_size + j]) == np.argmax(predictions[i*batch_size + j]):
                    matching_cases+=1
                eval_cases+=1
        return (100.0 * matching_cases) / eval_cases

    graph = tf.Graph()
    with graph.as_default():
        blstm = BLSTMNet(True, src2id, word_embeddings, word_embedding_dim, batch_size, seq_width, n_hidden)
        sense_models = {}
        is_first = True
        for lemma in lemma2synsets:
            is_first = False
            sense_models[lemma] = WSDNet(blstm.outputs, lemma, len(lemma2synsets[lemma]), n_hidden)
        #a = tf.trainable_variables()
        #trainer = Trainer()
    session = tf.Session(graph=graph)
    session.run(tf.initialize_all_variables())

    for step in range(training_iters):
        print 'EPOCH: %d' % step
        offset = (step * batch_size) % (len(data) - batch_size)
        input_data = new_batch(offset)
        run_epoch(session, sense_models, input_data, mode='train')

    '''
    # Run the tensorflow graph
    with tf.Session() as session:
        tf.initialize_all_variables().run()
        # Load pretrained embeddings
        print('Initialized')
        saver = tf.train.Saver()
        for step in range(training_iters):
            offset = (step * batch_size) % (len(training_data_list) - batch_size)
            batch_data, batch_labels, batch_seq_length, batch_weights_biases_id, batch_weights_biases_shape, \
            batch_training_points = new_batch(offset)
            feed_dict = {}
            for batch, batch_input in zip(tf_train_labels, batch_labels):
                for label, input in zip(batch, batch_input):
                    feed_dict.update({label : input})
            feed_dict.update({ tf_weight_bias : weight for tf_weight_bias, weight in
                               zip(tf_train_weights_biases, batch_weights_biases)})
            #for batch, batch_input in zip(tf_train_weights, batch_weights):
            #    for weights, input in zip(batch, batch_input):
            #        feed_dict.update({weights : input})
            #feed_dict.update({ tf_bias : bias for tf_bias, bias in zip(tf_train_biases, batch_biases)})
            #for batch, batch_input in zip(tf_train_biases, batch_biases):
            #    for bias, input in zip(batch, batch_input):
            #        feed_dict.update({bias : input})
            feed_dict.update({ tf_train_dataset : batch_data, tf_train_seq_length: batch_seq_length})
            #print feed_dict
            _, l, predictions = session.run(
              [optimizer_t, loss, train_prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
              print 'Minibatch loss at step ' + str(step) + ': ' + str(l)
              print 'Minibatch accuracy: ' + str(accuracy(predictions, batch_labels, batch_seq_length))
              if (args.save_path != "None" and step % 1000 == 0):
                saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=step)
                with open(os.path.join(args.save_path, 'src2id.pkl'), 'wb') as output:
                    pickle.dump(src2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2src.pkl'), 'wb') as output:
                    pickle.dump(id2src, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'target2id.pkl'), 'wb') as output:
                    pickle.dump(target2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2target.pkl'), 'wb') as output:
                    pickle.dump(id2target, output, pickle.HIGHEST_PROTOCOL)
    '''
