import argparse
import os
import pickle

import tensorflow as tf

import architectures
import misc

from data_ops import format_data, read_data, load_embeddings, evaluation


def run_epoch(session, model, inputs1, inputs2, sequence_lengths, labels_classif, labels_context, labels_pos, indices,
              pos_classifier, mode, wsd_method):
    """Runs one epoch of the neural model and returns a list of specified tensors

    Args:
        session: A tf.Session object in which the model is run
        model: An AbstractModel object, holds the parameters and logic of the model
        inputs1: A list of lists, the integer IDs for the inputs in the primary embedding model
        inputs2: A list of lists, the integer IDs for the inputs in the auxiliary embedding model, if in use
        sequence_lengths: A list of ints, the lengths of the individual sentences
        labels_classif: A list of (one-hot) arrays, the gold labels for the WSD classification method, if in use
        labels_context: A list of arrays, the "gold" embeddings for the context embedding WSD method, if in use
        labels_pos: A list of (one-hot) arrays, the gold labels for the POS classification method, if in use
        indices: A list of integers, indexes which words in the data are to be disambiguated
        pos_classifier: A boolean, indicates whether POS tagging should be carried out
        mode: A synset, indicates whether the epoch should be executed as: training, validation or evaluation
        wsd_method: A synset, indicates which model should be used: classification, context_embedding or multitask

    Returns:
        fetches: A list of the tensors to be retrieved from the network run

    """
    # feed_dict = {}
    # if mode != "evaluation":
    feed_dict = {model.train_inputs1: inputs1,
                 model.train_seq_lengths: sequence_lengths,
                 model.train_indices_wsd: indices,
                 }
    if wsd_method == "classification":
        feed_dict.update({model.train_labels_wsd: labels_classif})
    elif wsd_method == "context_embedding":
        feed_dict.update({model.train_labels_wsd: labels_context})
    elif wsd_method == "multitask":
        feed_dict.update({model.train_labels_wsd: labels_classif,
                          model.train_labels_wsd_context: labels_context})
    if len(inputs2) > 0:
        feed_dict.update({model.train_inputs2: inputs2})
    if pos_classifier is True:
        feed_dict.update({model.train_labels_pos: labels_pos})
    if mode == "train":
        ops = [model.train_op, model.cost, model.outputs_wsd, model.losses_wsd, model.logits_pos]
    elif mode == "validation":
        ops = [model.train_op, model.cost, model.outputs_wsd, model.losses_wsd, model.logits_pos,
               model.test_outputs_wsd, model.test_logits_pos]
    elif mode == "evaluation":
        ops = [model.outputs_wsd, model.logits_pos]
    fetches = session.run(ops, feed_dict=feed_dict)
    return fetches


if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0', description='Train or evaluate a neural WSD model.',
                                     fromfile_prefix_chars='@')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-embeddings1_path', dest='embeddings1_path', required=True,
                        help='The path to the pretrained model with the primary embeddings.')
    parser.add_argument('-embeddings2_path', dest='embeddings2_path', required=False,
                        help='The path to the pretrained model with the additional embeddings.')
    parser.add_argument('-embeddings1_case', dest='embeddings1_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings2_case', dest='embeddings2_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings1_dim', dest='embeddings1_dim', required=False, default=300,
                        help='Size of the primary embeddings.')
    parser.add_argument('-embeddings2_dim', dest='embeddings2_dim', required=False, default=300,
                        help='Size of the additional embeddings.')
    parser.add_argument('-embeddings1_input', dest='embeddings1_input', required=False, default="wordform",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-embeddings2_input', dest='embeddings2_input', required=False, default="lemma",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-keep_prob', dest='keep_prob', required=False, default="1",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.2,
                        help='How fast the network should learn.')
    parser.add_argument('-lexicon_path', dest='lexicon_path', required=False,
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-max_seq_length', dest='max_seq_length', required=False, default=63,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training, evaluation or application run? Options: train, evaluation  , "
                             "application")
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-pos_classifier', dest='pos_classifier', required=False, default="False",
                        help='Should the system also perform POS tagging? Available only with classification.')
    parser.add_argument('-pos_tagset', dest='pos_tagset', required=False, default="coarsegrained",
                        help='Whether the POS tags should be converted. Options are: coarsegrained, finegrained.')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved, or path to the folder with a saved model.')
    parser.add_argument('-sensekey2synset_path', dest='sensekey2synset_path', required=False,
                        help='Path to mapping between sense annotations in the corpus and synset IDs in WordNet.')
    parser.add_argument('-test_data_path', dest='test_data', required=False,
                        help='The path to the gold corpus used for testing.')
    parser.add_argument("-test_data_format", dest="test_data_format", required=False,
                        help="Specifies the format of the evaluation corpus. Options: naf, uef")
    parser.add_argument('-train_data_path', dest='train_data', required=False,
                        help='The path to the gold corpus used for training.')
    parser.add_argument("-train_data_format", dest="train_data_format", required=False,
                        help="Specifies the format of the training corpus. Options: naf, uef")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default=100001,
                        help='How many iterations the network should train for.')
    parser.add_argument('-wsd_classifier', dest='wsd_classifier', required=True,
                        help='Should the system perform WSD?')
    parser.add_argument('-wsd_method', dest='wsd_method', required=True,
                        help='Which method for WSD? Options: classification, context_embedding, multitask')

    ''' Get the parameters of the model from the command line '''
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    embeddings1_path = args.embeddings1_path
    embeddings1_case = args.embeddings1_case
    embeddings1_dim = int(args.embeddings1_dim)
    embeddings1_input = args.embeddings1_input
    embeddings2_path = args.embeddings2_path
    embeddings2_case = args.embeddings2_case
    if embeddings2_path is not None:
        embeddings2_dim = int(args.embeddings2_dim)
    else:
        embeddings2_dim = 0
    embeddings2_input = args.embeddings2_input
    keep_prob = float(args.keep_prob)
    learning_rate = float(args.learning_rate)
    lexicon_path = args.lexicon_path
    max_seq_length = int(args.max_seq_length)
    mode = args.mode
    n_hidden = int(args.n_hidden)
    n_hidden_layers = int(args.n_hidden_layers)
    pos_classifier = misc.str2bool(args.pos_classifier)
    pos_tagset = args.pos_tagset
    save_path = args.save_path
    sensekey2synset_path = args.sensekey2synset_path
    test_data_path = args.test_data
    test_data_format = args.test_data_format
    train_data_path = args.train_data
    train_data_format = args.train_data_format
    training_iterations = args.training_iterations
    wsd_classifier = misc.str2bool(args.wsd_classifier)
    wsd_method = args.wsd_method

    ''' Load the embedding model(s) that will be used '''
    embeddings1, emb1_src2id, emb1_id2src = load_embeddings.load(embeddings1_path)
    if embeddings2_path is not None:
        embeddings2, emb2_src2id, emb2_id2src = load_embeddings.load(embeddings2_path)
    else:
        embeddings2, emb2_src2id, emb2_id2src = [], None, None

    ''' Read data and auxiliary resource according to specified formats'''
    if train_data_format == "uef" or test_data_format == "uef":
        sensekey2synset = pickle.load(open(sensekey2synset_path, "rb"))
    lemma2synsets = read_data.get_wordnet_lexicon(lexicon_path)
    if train_data_format == "naf":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_naf(
            train_data_path, lemma2synsets, for_training=True, wsd_method=wsd_method, pos_tagset=pos_tagset)
    elif train_data_format == "uef":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = read_data.read_data_uef(
            train_data_path, sensekey2synset, lemma2synsets, for_training=True, wsd_method=wsd_method)
    if test_data_format == "naf":
        test_data, _, _, _, _ = read_data.read_data_naf(
            test_data_path, lemma2synsets, lemma2id=lemma2id, known_lemmas=known_lemmas,
            synset2id=synset2id, for_training=False, wsd_method=wsd_method, pos_tagset=pos_tagset)
    elif test_data_format == "uef":
        test_data, _, _, _, _ = read_data.read_data_uef(
            test_data_path, sensekey2synset, lemma2synsets, lemma2id=lemma2id,
            known_lemmas=known_lemmas, synset2id=synset2id, for_training=False, wsd_method=wsd_method)

    ''' Transform the test data into the input format readable by the neural models'''
    (test_inputs1,
     test_inputs2,
     test_sequence_lengths,
     test_labels_classif,
     test_labels_context,
     test_labels_pos,
     test_indices,
     test_target_lemmas,
     test_synsets_gold,
     test_pos_filters) = format_data.format_data(
        test_data, emb1_src2id, embeddings1_input, embeddings1_case, synset2id, max_seq_length, embeddings1,
        emb2_src2id, embeddings2_input, embeddings2_case, embeddings1_dim, pos_types, pos_classifier, wsd_method)

    ''' Initialize the neural model'''
    model = None
    if wsd_method == "classification":
        output_dimension = len(synset2id)
        model = architectures.classifier.ClassifierSoftmax(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_classif, wsd_classifier, pos_classifier, len(pos_types),
            test_labels_pos)
    elif wsd_method == "context_embedding":
        output_dimension = embeddings1_dim
        model = architectures.context_embedding.ContextEmbedder(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_context, wsd_classifier, pos_classifier, len(pos_types),
            test_labels_pos)
    elif wsd_method == "multitask":
        output_dimension = len(synset2id)
        model = architectures.multitask_wsd.MultitaskWSD(
            output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
            max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
            test_sequence_lengths, test_indices, test_labels_classif, test_labels_context, wsd_classifier,
            pos_classifier, len(pos_types), test_labels_pos)

    ''' Run training and/or evaluation'''
    session = tf.Session()
    saver = tf.train.Saver()
    if mode != "evaluation":
        model_path = os.path.join(save_path, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if wsd_method == "multitask":
            model_path_context = os.path.join(save_path, "model_context")
            if not os.path.exists(model_path_context):
                os.makedirs(model_path_context)
    if mode == "evaluation":
        with open(os.path.join(save_path, "checkpoint"), "r") as f:
            for line in f.readlines():
                if line.split()[0] == "model_checkpoint_path:":
                    model_checkpoint_path = line.split()[1].rstrip("\n")
                    model_checkpoint_path = model_checkpoint_path.strip("\"")
                    break
        saver.restore(session, model_checkpoint_path)
        # saver.restore(session, save_path)
        app_data = test_data
        match_wsd_classif_total, eval_wsd_classif_total, match_classif_wsd, eval_classif_wsd = [0, 0, 0, 0]
        match_wsd_context_total, eval_wsd_context_total, match_wsd_context, eval_wsd_context = [0, 0, 0, 0]
        match_pos_total, eval_pos_total, match_pos, eval_pos = [0, 0, 0, 0]
        for step in range(len(app_data) / batch_size + 1):
            offset = (step * batch_size) % (len(app_data))
            if offset + batch_size > len(app_data):
                buffer =  (offset + batch_size) - len(app_data)
                zero_element = ["UNK", "UNK", ".", ["unspecified"], [-1]]
                zero_sentence = batch_size * [zero_element]
                buffer_data = buffer * [zero_sentence]
                app_data.extend(buffer_data)
            (inputs1,
             inputs2,
             seq_lengths,
             labels_classif,
             labels_context,
             labels_pos,
             indices,
             target_lemmas,
             synsets_gold,
             pos_filters) = format_data.new_batch(
                offset, batch_size, app_data, emb1_src2id, embeddings1_input, embeddings1_case,
                synset2id, max_seq_length, embeddings1, emb2_src2id, embeddings2_input,
                embeddings2_case, embeddings1_dim, pos_types, pos_classifier, wsd_method)
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, keep_prob, pos_classifier, "evaluation", wsd_method)
            if wsd_method == "classification":
                _, _, [match_classif_wsd, eval_classif_wsd, match_pos, eval_pos] = evaluation.accuracy_classification(
                    fetches[0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, fetches[1], labels_pos)
            elif wsd_method == "context_embedding":
                _, [match_wsd_context, eval_wsd_context] = evaluation.accuracy_cosine_distance(
                    fetches[2], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
            elif wsd_method == "multitask":
                _, _, [match_classif_wsd, eval_classif_wsd, _, _] = evaluation.accuracy_classification(
                    fetches[0][0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, labels_pos)
                _, [match_wsd_context, eval_wsd_context] = evaluation.accuracy_cosine_distance(
                    fetches[0][1], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
            match_wsd_classif_total += match_classif_wsd
            eval_wsd_classif_total += eval_classif_wsd
            match_wsd_context_total += match_wsd_context
            eval_wsd_context_total += eval_wsd_context
            match_pos_total += match_pos
            eval_pos_total += eval_pos
        if wsd_method == "classification":
            print "Accuracy for WSD (CLASSIFICATION) is " + \
                  str((100.0 * match_wsd_classif_total) / eval_wsd_classif_total) + "%"
        elif wsd_method == "context_embedding":
            print "Accuracy for WSD (CONTEXT_EMBEDDING) is " + \
                  str((100.0 * match_wsd_context_total) / eval_wsd_context_total) + "%"
        elif wsd_method == "multitask":
            print "Accuracy for WSD (CLASSIFICATION) is " + \
                  str((100.0 * match_wsd_classif_total) / eval_wsd_classif_total) + "%"
            print "Accuracy for WSD (CONTEXT_EMBEDDING) is " + \
                  str((100.0 * match_wsd_context_total) / eval_wsd_context_total) + "%"
        if pos_classifier is True:
            print "Accuracy for POS tagging is " + \
                  str((100.0 * match_pos_total) / eval_pos_total) + "%"
        exit()
    else:
        init = tf.global_variables_initializer()
        feed_dict = {model.emb1_placeholder: embeddings1}
        if len(embeddings2) > 0:
            feed_dict.update({model.emb2_placeholder: embeddings2})
        session.run(init, feed_dict=feed_dict)
    print "Start of training"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    results = open(os.path.join(args.save_path, 'results.txt'), "a", 0)
    results.write(str(args) + '\n\n')
    batch_loss = 0
    best_accuracy_wsd, best_accuracy_context = 0.0, 0.0
    for step in range(training_iterations):
        offset = (step * batch_size) % (len(train_data) - batch_size)
        (inputs1,
         inputs2,
         seq_lengths,
         labels_classif,
         labels_context,
         labels_pos,
         indices,
         target_lemmas,
         synsets_gold,
         pos_filters) = format_data.new_batch(
            offset, batch_size, train_data, emb1_src2id, embeddings1_input, embeddings1_case,
            synset2id, max_seq_length, embeddings1, emb2_src2id, embeddings2_input,
            embeddings2_case, embeddings1_dim, pos_types, pos_classifier, wsd_method)
        test_accuracy_wsd, test_accuracy_context = 0.0, 0.0
        if step % 100 == 0:
            print "Step number " + str(step)
            results.write('EPOCH: %d' % step + '\n')
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, pos_classifier, "validation", wsd_method)
            if fetches[1] is not None:
                batch_loss += fetches[1]
            results.write('Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss / 100.0) + '\n')
            if wsd_method == "classification":
                minibatch_accuracy_wsd, minibatch_accuracy_pos, _ = evaluation.accuracy_classification(
                    fetches[2], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, fetches[4], labels_pos)
                test_accuracy_wsd, test_accuracy_pos, _ = evaluation.accuracy_classification(
                    fetches[5], test_target_lemmas, test_synsets_gold, test_pos_filters, synset2id, lemma2synsets,
                    known_lemmas, wsd_classifier, pos_classifier, fetches[6], test_labels_pos)
                if wsd_classifier is True:
                    results.write('Minibatch WSD accuracy: ' + str(minibatch_accuracy_wsd) + '\n')
                    results.write('Validation WSD accuracy: ' + str(test_accuracy_wsd) + '\n')
                if pos_classifier is True:
                    results.write('Minibatch POS tagging accuracy: ' + str(minibatch_accuracy_pos) + '\n')
                    results.write('Validation POS tagging accuracy: ' + str(test_accuracy_pos) + '\n')
            elif wsd_method == "context_embedding":
                minibatch_accuracy_wsd, _ = evaluation.accuracy_cosine_distance(
                    fetches[2], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
                test_accuracy_wsd, _ = evaluation.accuracy_cosine_distance(
                    fetches[5], test_target_lemmas, test_synsets_gold, test_pos_filters, lemma2synsets, embeddings1,
                    emb1_src2id)
                results.write('Minibatch WSD accuracy: ' + str(minibatch_accuracy_wsd) + '\n')
                results.write('Validation WSD accuracy: ' + str(test_accuracy_wsd) + '\n')
            elif wsd_method == "multitask":
                minibatch_accuracy_wsd, _, _ = evaluation.accuracy_classification(
                    fetches[2][0], target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                    wsd_classifier, pos_classifier, labels_pos)
                test_accuracy_wsd, _, _ = evaluation.accuracy_classification(
                    fetches[5][0], test_target_lemmas, test_synsets_gold, test_pos_filters, synset2id, lemma2synsets,
                    known_lemmas, wsd_classifier, pos_classifier, test_labels_pos)
                minibatch_accuracy_context, _ = evaluation.accuracy_cosine_distance(
                    fetches[2][1], target_lemmas, synsets_gold, pos_filters, lemma2synsets, embeddings1, emb1_src2id)
                test_accuracy_context, _ = evaluation.accuracy_cosine_distance(
                    fetches[5][1], test_target_lemmas, test_synsets_gold, test_pos_filters, lemma2synsets, embeddings1,
                    emb1_src2id)
                results.write('Minibatch WSD accuracy (CLASSIFICATION): ' + str(minibatch_accuracy_wsd) + '\n')
                results.write('Validation WSD accuracy (CLASSIFICATION): ' + str(test_accuracy_wsd) + '\n')
                results.write('Minibatch WSD accuracy (CONTEXT_EMBEDDING): ' + str(minibatch_accuracy_context) + '\n')
                results.write('Validation WSD accuracy (CONTEXT_EMBEDDING): ' + str(test_accuracy_context) + '\n')
            print "Validation accuracy: " + str(test_accuracy_wsd)
            if wsd_method == "multitask":
                print "Validation accuracy (CONTEXT_EMBEDDING): " + str(test_accuracy_context)
            batch_loss = 0.0
        else:
            fetches = run_epoch(session, model, inputs1, inputs2, seq_lengths, labels_classif, labels_context,
                                labels_pos, indices, pos_classifier, "train", wsd_method)
            if fetches[1] is not None:
                batch_loss += fetches[1]
        if test_accuracy_wsd > best_accuracy_wsd:
            best_accuracy_wsd = test_accuracy_wsd
        if wsd_method == "multitask" and test_accuracy_context > best_accuracy_context:
            best_accuracy_context = test_accuracy_context
        if args.save_path is not None:
            if step == 100 or step > 100 and test_accuracy_wsd == best_accuracy_wsd:
                for file in os.listdir(model_path):
                    os.remove(os.path.join(model_path, file))
                saver.save(session, os.path.join(args.save_path, "model/model.ckpt"), global_step=step)
            if wsd_method == "multitask" and \
                    (step == 100 or step > 100 and test_accuracy_context == best_accuracy_context):
                for file in os.listdir(model_path_context):
                    os.remove(os.path.join(model_path_context, file))
                saver.save(session, os.path.join(args.save_path, "model_context/model_context.ckpt"), global_step=step)
    results.write('\n\n\n' + 'Best result is: ' + str(best_accuracy_wsd))
    if wsd_method == "multitask":
        results.write('\n\n\n' + 'Best result (CONTEXT_EMBEDDING) is: ' + str(best_accuracy_context))
    results.close()
