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
        self.max_seq_length = max_seq_length
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.wsd_classifier = wsd_classifier
        self.pos_classifier = pos_classifier
        self.emb1_placeholder = tf.placeholder(tf.float32, shape=[vocab_size1, emb1_dim], name="emb1_placeholder")
        self.embeddings1 = tf.Variable(self.emb1_placeholder)
        self.set_embeddings1 = tf.assign(self.embeddings1, self.emb1_placeholder, validate_shape=False)
        if vocab_size2 > 0:
            self.emb2_placeholder = tf.placeholder(tf.float32, shape=[vocab_size2, emb2_dim], name="emb2_placeholder")
            self.embeddings2 = tf.Variable(self.emb2_placeholder)
            self.set_embeddings2 = tf.assign(self.embeddings2, self.emb2_placeholder, validate_shape=False)
        if wsd_classifier is True:
            self.weights_wsd = tf.get_variable(name="softmax_wsd-w", shape=[2 * 2 * n_hidden, output_dim],
                                               dtype=tf.float32)
            self.biases_wsd = tf.get_variable(name="softmax_wsd-b", shape=[output_dim], dtype=tf.float32)
            self.train_labels_wsd = tf.placeholder(name="train_labels_wsd", dtype=test_labels_wsd.dtype, shape=[None, output_dim])
            self.train_indices_wsd = tf.placeholder(name="train_indices_wsd", dtype=tf.int32, shape=[None])
            self.test_labels_wsd = tf.constant(test_labels_wsd, test_labels_wsd.dtype, name="test_labels_wsd")
            self.test_indices_wsd = tf.constant(test_indices_wsd, tf.int32, name="test_indices_wsd")
        else:
            self.weights_wsd = None
            self.biases_wsd = None
            self.train_labels_wsd = None
            self.train_indices_wsd = None
            self.test_indices_wsd = None
            self.test_labels_wsd = None
        self.train_inputs1 = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name="train_inputs1")
        self.train_inputs2 = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name="train_inputs2")
        self.train_seq_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="train_seq_lengths")
        if pos_classifier is True:
            self.weights_pos = tf.get_variable(name="softmax_pos-w", shape=[2*n_hidden, pos_classes], dtype=tf.float32)
            self.biases_pos = tf.get_variable(name="softmax_pos-b", shape=[pos_classes], dtype=tf.float32)
            self.train_labels_pos = tf.placeholder(name="pos_labels", shape=[None, pos_classes], dtype=tf.int32)
            self.test_labels_pos = tf.constant(test_labels_pos, tf.int32, name="test_labels_pos")
        else:
            self.weights_pos = None
            self.biases_pos = None
            self.train_labels_pos = None
            self.test_labels_pos = None
        self.test_inputs1 = tf.constant(test_inputs1, tf.int32, name="test_inputs1")
        if vocab_size2 > 0:
            self.test_inputs2 = tf.constant(test_inputs2, tf.int32, name="test_inputs2")
        self.test_seq_lengths = tf.constant(test_seq_lengths, tf.int32, name="test_seq_lengths")
        self.attn_param = tf.get_variable(name="attention_param_vector", shape=[2*n_hidden],
                                          dtype=tf.float32)

    def run_neural_model(self):
        """Runs the model: embeds the inputs, calculates recurrences and performs classification/regression"""
        if self.vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.train_inputs1, self.train_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.train_inputs1)
        self.cost, self.outputs_wsd, self.losses_wsd, self.logits_pos, self.attn_weights = self.biRNN_WSD\
            (True, self.n_hidden_layers, self.n_hidden, self.train_seq_lengths, embedded_inputs, self.batch_size,
             self.wsd_classifier, self.pos_classifier)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        if self.vocab_size2 > 0:
            embedded_inputs = self.embed_inputs(self.test_inputs1, self.test_inputs2)
        else:
            embedded_inputs = self.embed_inputs(self.test_inputs1)
        tf.get_variable_scope().reuse_variables()
        _, self.test_outputs_wsd, _, self.test_logits_pos, _ = self.biRNN_WSD\
            (False, self.n_hidden_layers, self.n_hidden, self.test_seq_lengths, embedded_inputs, tf.shape(self.test_inputs1)[0],
             self.wsd_classifier, self.pos_classifier)


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


    def biRNN_WSD(self, is_training, n_hidden_layers, n_hidden, seq_lengths, embedded_inputs,  batch_size, wsd_classifier=True,
                  pos_classifier=False):
        """Bi-directional long short-term memory (Bi-LSTM) layer

        Args:
            is_training: A boolean, indicates whether the output of the layer will be used for training
            n_hidden_layers: An int, the number of Bi-LSTMs to be initialized
            n_hidden: An int, the number of neurons per layer in the LSTMs
            seq_lengths: A tensor of ints, the lengths of the individual sentences
            embedded_inputs: A tensor of floats, the inputs to the RNN layer
            wsd_classifier: A bool, indicates whether WSD should be learned by the model
            pos_classifier: A bool, indicates whether POS tagging should be learned by the model

        Returns:
            cost: A float, the loss against which the model is to be trained
            outputs_wsd: A tensor of floats, the output layer vector produced by the model for the WSD task
            losses_wsd: A tensor, the losses per disambiguated word
            logits_pos: A tensor of floats, the output layer vector produced by the model for the POS tagging task

        """
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            initializer = tf.random_uniform_initializer(-1, 1)

            def lstm_cell():
                lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
                if is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper\
                        (lstm_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                return lstm_cell

            fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
            bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(n_hidden_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_multicell, bw_multicell, embedded_inputs,
                                                             dtype="float32", sequence_length=seq_lengths)
            rnn_outputs = tf.concat(rnn_outputs, 2)
            # rnn_outputs = tf.layers.dropout(rnn_outputs, rate=(1-self.keep_prob), training=is_training)
            # word_project = tf.layers.dense(embedded_inputs, units=2 * n_hidden, use_bias=False)
            # rnn_outputs = rnn_outputs + word_project
            # outputs = self.layer_normalize(rnn_outputs)
            # attn_outputs = self.multi_head_attention(outputs, outputs, 1, None, drop_rate=(1-self.keep_prob),
            #                                         is_train=is_training)
            rnn_outputs_flipped = tf.transpose(rnn_outputs, [0, 2, 1])
            attn_weights = tf.matmul(tf.tile(tf.reshape(self.attn_param, [1, 1, 2*n_hidden]), [batch_size, 1, 1]),
                                     tf.nn.tanh(rnn_outputs_flipped))
            attn_weights = tf.nn.softmax(attn_weights)
            ctx_vector = tf.matmul(rnn_outputs_flipped, tf.transpose(attn_weights, [0, 2, 1]))
            ctx_vector = tf.tile(tf.transpose(ctx_vector, [0, 2, 1]), [1, self.max_seq_length, 1])
            outputs = tf.concat((rnn_outputs, ctx_vector), axis=2)
            # attn_outputs = attn_outputs + outputs
            #TODO non-linearity
            #outputs = self.layer_normalize(attn_outputs)
            scope.reuse_variables()
            outputs = tf.reshape(outputs, [-1, 2 * 2 * n_hidden])
            logits_pos, losses_pos, cost_pos = [], [], 0.0
            if pos_classifier is True:
                logits_pos, losses_pos, cost_pos = self.output_layer(outputs, is_training, classif_type="pos")
            outputs_wsd, losses_wsd, cost_wsd = [], [], 0.0
            if wsd_classifier is True:
                outputs_wsd, losses_wsd, cost_wsd = self.output_layer(outputs, is_training, classif_type="wsd")
            cost = cost_wsd + cost_pos
        return cost, outputs_wsd, losses_wsd, logits_pos, attn_weights


    def output_layer(self, rnn_outputs, is_training, classif_type):
        """Calculates the output of the network, specific to the concrete types of models"""
        raise Exception ("Not implemented!")

    def layer_normalize(self, inputs, epsilon=1e-8, scope=None):
        with tf.variable_scope(scope or "layer_norm"):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = tf.add(tf.multiply(gamma, normalized), beta)
            return outputs

    def multi_head_attention(self, queries, keys, num_heads, attention_size, drop_rate=0.0, is_train=True, reuse=None,
                             scope=None):
        # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
        with tf.variable_scope(scope or "multi_head_attention", reuse=reuse):
            if attention_size is None:
                attention_size = queries.get_shape().as_list()[-1]
            # linear projections, shape=(batch_size, max_time, attention_size)
            query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_project")
            key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_project")
            value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_project")
            # split and concatenation, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
            query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
            key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
            value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
            # multiplication
            attn_outs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
            # scale
            attn_outs = attn_outs / (key_.get_shape().as_list()[-1] ** 0.5)
            # key masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # shape=(batch_size, max_time)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # shape=(batch_size * num_heads, max_time)
            # shape=(batch_size * num_heads, max_time, max_time)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
            paddings = tf.ones_like(attn_outs) * (-2 ** 32 + 1)
            # shape=(batch_size, max_time, attention_size)
            attn_outs = tf.where(tf.equal(key_masks, 0), paddings, attn_outs)
            # activation
            attn_outs = tf.nn.softmax(attn_outs)
            # query masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            attn_outs *= query_masks
            # dropout
            attn_outs = tf.layers.dropout(attn_outs, rate=drop_rate, training=is_train)
            # weighted sum
            outputs = tf.matmul(attn_outs, value_)
            # restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            outputs += queries  # residual connection
            outputs = self.layer_normalize(outputs)
            return outputs
