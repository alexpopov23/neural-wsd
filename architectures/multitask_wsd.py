import tensorflow as tf

from abstract_model import AbstractModel


class MultitaskWSD(AbstractModel):

    def __init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length, n_hidden,
                 n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2, test_seq_lengths,
                 test_indices_wsd, test_labels_wsd, test_labels_wsd_context, wsd_classifier=True, pos_classifier=False, pos_classes=0,
                 test_pos_labels=None):
        AbstractModel.__init__(self, output_dim, vocab_size1, emb1_dim, vocab_size2, emb2_dim, batch_size, max_seq_length,
                               n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
                               test_seq_lengths, test_indices_wsd, test_labels_wsd, wsd_classifier, pos_classifier,
                               pos_classes, test_pos_labels)
        self.weights_wsd_context = tf.get_variable(name="context_wsd-w", shape=[2*n_hidden, emb1_dim], dtype=tf.float32)
        self.biases_wsd_context = tf.get_variable(name="context_wsd-b", shape=[emb1_dim], dtype=tf.float32)
        self.train_labels_wsd_context = tf.placeholder(tf.float32, shape=[None, emb1_dim],
                                                       name="train_labels_context_embedding")
        self.test_labels_wsd_context = tf.constant(test_labels_wsd_context, tf.float32)
        self.run_neural_model()


    def output_layer(self, rnn_outputs, labels_classif, labels_context, indices, classif_type="wsd"):
        target_outputs = tf.gather(rnn_outputs, indices)
        if classif_type != "wsd":
            raise Exception ("Classification for tasks other than WSD not implemented in this model!")
        logits_classif = tf.matmul(target_outputs, self.weights_wsd) + self.biases_wsd
        losses_classif = tf.nn.softmax_cross_entropy_with_logits(logits=logits_classif, labels=labels_classif)
        cost_classif = tf.reduce_mean(losses_classif)
        output_embeddings = tf.matmul(target_outputs, self.weights_wsd_context) + self.biases_wsd_context
        losses_context = (labels_context - output_embeddings) ** 2
        cost_context = tf.reduce_mean(losses_context)
        cost_wsd = cost_classif + cost_context
        return (logits_classif, output_embeddings), (losses_classif, losses_context), cost_wsd