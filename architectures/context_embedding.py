import tensorflow as tf

from abstract_model import AbstractModel


class ContextEmbedder(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        """See docstring for AbstractModel"""
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, labels, indices, classif_type="wsd"):
        """Output layer for the context embedding model.

        Resizes the RNN output to the size of the vector space model used and obtains a least squares
        loss per each disambiguation case. A mean value of the losses is passed along.

        Args:
            rnn_outputs: A tensor of floats, the output of the RNN layer
            labels: A tensor of floats, the gold data synset embeddings
            indices: A tensor of ints, indicates which words should be disambiguated
            classif_type: A string, the kind of classification to be carried out (only WSD implemented in this case)

        Returns:
            output_embeddings: A tensor of floats, the context embeddings used for the least squares computation
            losses: A tensor of floats, the computed losses for the individual choices
            cost_wsd: A float, the mean loss for the whole input

        """
        target_outputs = tf.gather(rnn_outputs, indices)
        if classif_type != "wsd":
            raise Exception ("Classification for tasks other than WSD not implemented in this model!")
        output_embeddings = tf.matmul(target_outputs, self.weights_wsd) + self.biases_wsd
        losses = (labels - output_embeddings) ** 2
        cost_wsd = tf.reduce_mean(losses)
        return output_embeddings, losses, cost_wsd