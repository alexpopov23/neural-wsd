import tensorflow as tf

from abstract_model import AbstractModel


class MultitaskWSD(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels_wsd, test_labels_wsd_context, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        """See docstring for AbstractModel for most of the parameters

        Additional args:
            test_labels_wsd_context: An array of floats, the gold data embeddings for the embedding pathway

        """
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels_wsd, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.weights_wsd_context = tf.get_variable(name="context_wsd-w", shape=[2*n_hidden, emb1_dim], dtype=tf.float32)
        self.biases_wsd_context = tf.get_variable(name="context_wsd-b", shape=[emb1_dim], dtype=tf.float32)
        self.train_labels_wsd_context = tf.placeholder(tf.float32, shape=[None, emb1_dim],
                                                       name="train_labels_wsd_context")
        # self.train_labels_wsd = (self.train_labels_wsd, self.train_labels_wsd_context)
        self.test_labels_wsd_context = tf.constant(test_labels_wsd_context, tf.float32)
        # self.test_labels_wsd = (self.test_labels_wsd, self.test_labels_wsd_context)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, is_training, classif_type="wsd"):
        """Output layer for the multitask WSD model.

        Resizes the RNN output to two separate vectors -- one the size of the output vocabulary used for
        classification and the other the size of the vector space model used for context embedding.
        Then obtains cross entropy losses and least squares losses per each disambiguation case.
        The combined loss is a sum of the mean losses per each of the two output layers.

        Args:
            rnn_outputs: A tensor of floats, the output of the RNN layer
            is_training: A bool, indicating whether the function is in "train" or "test" mode
            classif_type: A string, the kind of classification to be carried out (only WSD implemented in this case)

        Returns:
            output_embeddings: A tuple of tensors: logits for classification and context embeddings for regression
            losses: A tuple of tensors, each contains the losses per the two output layers
            cost_wsd: A float, the combined loss for the two output layers

        """
        if classif_type != "wsd":
            raise Exception ("Classification for tasks other than WSD not implemented in this model!")
        if is_training is True:
            indices = self.train_indices_wsd
            labels_classif = self.train_labels_wsd
            labels_context = self.train_labels_wsd_context
        else:
            indices = self.test_indices_wsd
            labels_classif = self.test_labels_wsd
            labels_context = self.test_labels_wsd_context
        target_outputs = tf.gather(rnn_outputs, indices)
        logits_classif = tf.matmul(target_outputs, self.weights_wsd) + self.biases_wsd
        losses_classif = tf.nn.softmax_cross_entropy_with_logits(logits=logits_classif, labels=labels_classif)
        cost_classif = tf.reduce_mean(losses_classif)
        output_embeddings = tf.matmul(target_outputs, self.weights_wsd_context) + self.biases_wsd_context
        losses_context = (labels_context - output_embeddings) ** 2
        cost_context = tf.reduce_mean(losses_context)
        cost_wsd = cost_classif + cost_context
        return (logits_classif, output_embeddings), (losses_classif, losses_context), cost_wsd