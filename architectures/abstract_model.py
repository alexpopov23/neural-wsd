import tensorflow as tf


class AbstractModel:
    """The basis for the neural models, provides the common input, embedding and RNN layers"""

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels_wsd, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_labels_pos=None):
        """Initializes the model

        Args:
            output_dim: An int, this is the size of the output layer, on which classification/regression is computed
            vocab_size1: An int, the number of words in the primary vector space model (VSM)
            emb1_dim: An int, the dimensionality of the vectors in the primary VSM
            vocab_size2: An int, the number of words in the secondary VSM, if one is provided
            emb2_dim: An int, the dimensionality of the vectors in the secondary VSM
            batch_size: An int, the size of the training mini-batches
            max_seq_length: The maximum length of the data sequences (used in LSTM to save computational resources)
            n_hidden: An int, the size of the individual layers in the LSTMs
            n_hidden_layers: An int, the depth of the Bi-LSTM layer
            learning_rate: A float, the rate of learning
            keep_prob: A float, the probability of keeping the activity of a neuron (dropout)
            test_inputs1: An array of ints, the embedding IDs for each word in the test sentences
            test_inputs2: An array of ints, (optional) auxiliary embedding IDs for each word in the test sentences
            test_seq_lengths: An array of ints, indicates the length of each sentence in the test data
            test_indices_wsd: An array of ints, indicates which words in the test data are to be disambiguated
            test_labels_wsd: An array, provides the gold data against which the model can be compared
            wsd_classifier: A bool, indicates whether WSD should be learned by the model
            pos_classifier: A bool, indicates whether POS tagging should be learned by the model
            pos_classes: An int, the number of POS classes found in the training data
            test_labels_pos: An array, the gold testing data for the POS task

        """
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.emb1_dim = emb1_dim
        self.emb2_dim = emb2_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.wsd_classifier = wsd_classifier
        self.pos_classifier = pos_classifier
        self.emb1_placeholder = tf.placeholder(tf.float32, shape=[vocab_size1, emb1_dim])
        self.embeddings1 = tf.Variable(self.emb1_placeholder)
        self.set_embeddings1 = tf.assign(self.embeddings1, self.emb1_placeholder, validate_shape=False)
        if vocab_size2 > 0:
            self.emb2_placeholder = tf.placeholder(tf.float32, shape=[vocab_size2, emb2_dim])
            self.embeddings2 = tf.Variable(self.emb2_placeholder)
            self.set_embeddings2 = tf.assign(self.embeddings2, self.emb2_placeholder, validate_shape=False)
        if wsd_classifier is True:
            self.weights_wsd = tf.get_variable(name="softmax_wsd-w", shape=[2 * n_hidden, output_dim],
                                               dtype=tf.float32)
            self.biases_wsd = tf.get_variable(name="softmax_wsd-b", shape=[output_dim], dtype=tf.float32)
            self.train_labels_wsd = tf.placeholder(dtype=test_labels_wsd.dtype, shape=[None, output_dim])
            self.train_indices_wsd = tf.placeholder(dtype=tf.int32, shape=[None])
            self.test_labels_wsd = tf.constant(test_labels_wsd, test_labels_wsd.dtype)
            self.test_indices_wsd = tf.constant(test_indices_wsd, tf.int32)
        else:
            self.weights_wsd = None
            self.biases_wsd = None
            self.train_labels_wsd = None
            self.train_indices_wsd = None
            self.test_indices_wsd = None
            self.test_labels_wsd = None
        self.train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length])
        self.train_inputs2 = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        if pos_classifier is True:
            self.weights_pos = tf.get_variable(name="softmax_pos-w", shape=[2*n_hidden, pos_classes], dtype=tf.float32)
            self.biases_pos = tf.get_variable(name="softmax_pos-b", shape=[pos_classes], dtype=tf.float32)
            self.train_labels_pos = tf.placeholder(name="pos_labels", shape=[None, pos_classes], dtype=tf.int32)
            self.test_labels_pos = tf.constant(test_labels_pos, tf.int32)
        else:
            self.weights_pos = None
            self.biases_pos = None
            self.train_labels_pos = None
            self.test_labels_pos = None
        self.test_inputs1 = tf.constant(test_inputs1, tf.int32)
        if vocab_size2 > 0:
            self.test_inputs2 = tf.constant(test_inputs2, tf.int32)
        self.test_seq_lengths = tf.constant(test_seq_lengths, tf.int32)

    def run_neural_model(self):
        """Runs the model: embeds the inputs, calculates recurrences and performs classification/regression"""
        if self.vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.train_inputs1, self.train_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.train_inputs1)
        self.cost, self.outputs_wsd, self.losses_wsd, self.logits_pos = \
            self.biRNN_WSD(True, self.n_hidden_layers, self.n_hidden, self.train_seq_lengths, self.train_indices_wsd,
                           self.train_labels_wsd, embedded_inputs, self.wsd_classifier, self.pos_classifier,
                           self.train_labels_pos)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        if self.vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.test_inputs1, self.test_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.test_inputs1)
        tf.get_variable_scope().reuse_variables()
        _, self.test_outputs_wsd, _, self.test_logits_pos = \
            self.biRNN_WSD(False, self.n_hidden_layers, self.n_hidden, self.test_seq_lengths, self.test_indices_wsd,
                           self.test_labels_wsd, embedded_inputs, self.wsd_classifier, self.pos_classifier,
                           self.test_labels_pos)


    def embed_inputs(self, inputs1, inputs2=None):
        """Takes one or two input sequences (of integer IDs) and embeds them in the resepective VSMs.

        If two models are used, the embeddings from the separate VSMs are concatenated in single vectors.

        Args:
            inputs1: An array of integer IDs (primary VSM)
            inputs2: An array of integer IDs (secondary VSM)

        Returns:
            embedded_inputs: An array of floats, the embedding vector

        """
        embedded_inputs = tf.nn.embedding_lookup(self.embeddings1, inputs1)
        if inputs2 is not None:
            embedded_inputs2 = tf.nn.embedding_lookup(self.embeddings2, inputs2)
            embedded_inputs = tf.concat([embedded_inputs, embedded_inputs2], 2)
        return embedded_inputs


    def biRNN_WSD(self, is_training, n_hidden_layers, n_hidden, seq_lengths, indices, labels, embedded_inputs,
                  wsd_classifier=True, pos_classifier=False, labels_pos=None):
        """Bi-directional long short-term memory (Bi-LSTM) layer

        Args:
            is_training: A boolean, indicates whether the output of the layer will be used for training
            n_hidden_layers: An int, the number of Bi-LSTMs to be initialized
            n_hidden: An int, the number of neurons per layer in the LSTMs
            seq_lengths: A tensor of ints, the lengths of the individual sentences
            indices: A tensor of ints, the words in the sentences that should be disambiguated
            labels: A tensor of ints or floats, the gold data
            embedded_inputs: A tensor of floats, the inputs to the RNN layer
            wsd_classifier: A bool, indicates whether WSD should be learned by the model
            pos_classifier: A bool, indicates whether POS tagging should be learned by the model
            labels_pos: A tensor of ints, the gold data for the POS tagging task

        Returns:
            cost: A float, the loss against which the model is to be trained
            outputs_wsd: A tensor of floats, the output layer vector produced by the model for the WSD task
            losses_wsd: A tensor, the losses per disambiguated word
            logits_pos: A tensor of floats, the output layer vector produced by the model for the POS tagging task

        """
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            initializer = tf.random_uniform_initializer(-1, 1)

            def lstm_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, initializer=initializer)
                if is_training:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper\
                        (lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                return lstm_cell

            fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
            bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs,
                                                             dtype="float32", sequence_length=seq_lengths)
            rnn_outputs = tf.concat(rnn_outputs, 2)
            scope.reuse_variables()
            rnn_outputs = tf.reshape(rnn_outputs, [-1, 2 * n_hidden])
            logits_pos, losses_pos, cost_pos = [], [], 0.0
            if pos_classifier is True:
                logits_pos, losses_pos, cost_pos = self.output_layer(rnn_outputs, labels_pos, classif_type="pos")
            outputs_wsd, losses_wsd, cost_wsd = [], [], 0.0
            if wsd_classifier is True:
                outputs_wsd, losses_wsd, cost_wsd = self.output_layer(rnn_outputs, labels, indices, classif_type="wsd")
            cost = cost_wsd + cost_pos
        return cost, outputs_wsd, losses_wsd, logits_pos


    def output_layer(self, rnn_outputs, labels, indices=None):
        """Calculates the output of the network, specific to the concrete types of models"""
        raise Exception ("Not implemented!")

