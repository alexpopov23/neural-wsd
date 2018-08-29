import tensorflow as tf

from abstract_model import AbstractModel


class ClassifierSoftmax(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, labels, indices=None, classif_type="wsd"):
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