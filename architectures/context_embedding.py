import tensorflow as tf

from abstract_model import AbstractModel


class ContextEmbedder(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, labels, indices, classif_type="wsd"):
        target_outputs = tf.gather(rnn_outputs, indices)
        if classif_type != "wsd":
            raise Exception ("Classification for tasks other than WSD not implemented in this model!")
        output_embeddings = tf.matmul(target_outputs, self.weights_wsd) + self.biases_wsd
        losses = (labels - output_embeddings) ** 2
        cost_wsd = tf.reduce_mean(losses)
        return output_embeddings, losses, cost_wsd