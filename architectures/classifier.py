import tensorflow as tf


class ClassifierSoftmax:

    def __init__(self, synset2id, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, seq_lengths, n_hidden,
                 n_hidden_layers, learning_rate, test_inputs1, test_inputs2, test_seq_lengths, test_indices, test_labels,
                 wsd_classifier=True, pos_classifier=False, pos_classes=0, test_pos_labels=None):
        self.emb1_placeholder = tf.placeholder(tf.float32, shape=[vocab_size1, emb1_dim])
        self.embeddings1 = tf.Variable(self.emb1_placeholder)
        self.set_embeddings1 = tf.assign(self.embeddings1, self.emb1_placeholder, validate_shape=False)
        if vocab_size2 > 0:
            self.emb2_placeholder = tf.placeholder(tf.float32, shape=[vocab_size2, emb2_dim])
            self.embeddings2 = tf.Variable(self.emb2_placeholder)
            self.set_embeddings2 = tf.assign(self.embeddings2, self.emb2_placeholder, validate_shape=False)
        if wsd_classifier is True:
            self.weights_wsd = tf.get_variable(name="softmax_wsd-w", shape=[2*n_hidden, len(synset2id)], dtype=tf.float32)
            self.biases_wsd = tf.get_variable(name="softmax_wsd-b", shape=[len(synset2id)], dtype=tf.float32)
            self.train_labels_wsd = tf.placeholder(tf.int32, shape=[None, len(synset2id)])
            self.train_indices_wsd = tf.placeholder(tf.int32, shape=[None])
        else:
            self.weights_wsd = None
            self.biases_wsd = None
            self.train_labels_wsd = None
            self.train_indices_wsd = None
        self.train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size, seq_lengths])
        self.train_inputs2 = tf.placeholder(tf.int32, shape=[batch_size, seq_lengths])
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
        if pos_classifier is True:
            self.weights_pos = tf.get_variable(name="softmax_pos-w", shape=[2*n_hidden, pos_classes], dtype=tf.float32)
            self.biases_pos = tf.get_variable(name="softmax_pos-b", shape=[pos_classes], dtype=tf.float32)
            self.train_labels_pos = tf.placeholder(name="pos_labels", shape=[None, pos_classes], dtype=tf.int32)
            self.test_labels_pos = tf.constant(test_pos_labels, tf.int32)
        else:
            self.weights_pos = None
            self.biases_pos = None
            self.train_labels_pos = None
            self.test_labels_pos = None
        self.test_inputs1 = tf.constant(test_inputs1, tf.int32)
        if vocab_size2 > 0:
            self.test_inputs2 = tf.constant(test_inputs2, tf.int32)
        self.test_seq_lengths = tf.constant(test_seq_lengths, tf.int32)
        self.place = tf.placeholder(tf.int32, shape=test_labels.shape)
        self.test_labels = tf.Variable(self.place)
        self.test_indices = tf.constant(test_indices, tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        # if two sets of embeddings are passed, then concatenate them together
        if vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.train_inputs1, self.train_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.train_inputs1)
        self.cost, self.logits, self.losses, self.logits_pos = \
            self.biRNN_WSD(True, n_hidden_layers, n_hidden, self.train_seq_lengths, self.train_indices_wsd,
                           self.train_indices_wsd, self.train_labels_wsd, embedded_inputs, wsd_classifier,
                           pos_classifier, self.train_labels_pos)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        if vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.test_inputs1, self.test_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.test_inputs1)
        tf.get_variable_scope().reuse_variables()
        _, self.test_logits, _, self.test_logits_pos = \
            self.biRNN_WSD(False, n_hidden_layers, n_hidden, self.test_seq_lengths, self.test_indices_wsd,
                           self.test_indices_wsd, self.test_labels_wsd, embedded_inputs, wsd_classifier,
                           pos_classifier, self.test_labels_pos)


    def embed_inputs(self, inputs1, inputs2=None):

        embedded_inputs = tf.nn.embedding_lookup(self.embeddings, inputs1)
        if inputs2 is not None:
            embedded_inputs2 = tf.nn.embedding_lookup(self.embeddings_lemmas, inputs2)
            embedded_inputs = tf.concat([embedded_inputs, embedded_inputs2], 2)

        return embedded_inputs


    def biRNN_WSD(self, is_training, n_hidden_layers, n_hidden, seq_lengths, indices, labels, embedded_inputs,
                  wsd_classifier=True, pos_classifier=False, labels_pos=None):

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            initializer = tf.random_uniform_initializer(-1, 1)

            # TODO: Use state_is_tuple=True
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
            logits_pos, cost_pos = [], 0.0
            if pos_classifier is True:
                logits_pos = tf.matmul(rnn_outputs, self.weights_pos) + self.biases_pos
                losses_pos = tf.nn.softmax_cross_entropy_with_logits(logits=logits_pos, labels=labels_pos)
                cost_pos = tf.reduce_mean(losses_pos)
            logits, losses, cost_wsd = [], [], 0.0
            if wsd_classifier is True:
                target_outputs = tf.gather(rnn_outputs, indices)
                logits = tf.matmul(target_outputs, self.weights_wsd) + self.biases_wsd
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cost_wsd = tf.reduce_mean(losses)
            cost = cost_wsd + cost_pos
        return cost, logits, losses, logits_pos