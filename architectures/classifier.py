import tensorflow as tf

from abstract_model import AbstractModel


class ClassifierSoftmax(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        """See docstring for AbstratModel"""
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, labels, indices=None, classif_type="wsd"):
        """Output layer for the classifier model.

        Resizes the RNN output to the size of the output vocabulary, obtains a softmax probability distribution
        and calculates a cross entropy loss per each choice. A mean value of the losses is passed along.

        Args:
            rnn_outputs: A tensor of floats, the output of the RNN layer
            labels: A tensor of ints, the gold data one-hot vectors
            indices: A tensor of ints, indicates which words should be disambiguated
            classif_type: A string, the kind of classification to be carried out: WSD or POS tagging

        Returns:
            logits: A tensor of floats, the logits used for the softmax function
            losses: A tensor of floats, the computed losses for the individual choices
            cost_wsd: A float, the mean loss for the whole input

        """
        if indices is not None:
            target_outputs = tf.gather(rnn_outputs, indices)
        else:
            target_outputs = rnn_outputs
        if classif_type == "wsd":
            weights, biases = self.weights_wsd, self.biases_wsd
        elif classif_type == "pos":
            weights, biases = self.weights_pos, self.biases_pos
        logits = tf.matmul(target_outputs, weights) + biases
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cost_wsd = tf.reduce_mean(losses)
        return logits, losses, cost_wsd