import argparse
import pickle

import tensorflow as tf

import architectures
import format_data
import load_embeddings
import misc
import read_data


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
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, mixedcase')
    parser.add_argument('-embeddings2_case', dest='embeddings2_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, mixedcase')
    parser.add_argument('-embeddings1_dim', dest='embeddings1_dim', required=False, default=300,
                        help='Size of the primary embeddings.')
    parser.add_argument('-embeddings2_dim', dest='embeddings2_dim', required=False, default=0,
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
                        help="Is this is a training, evaluation or application run? Options: train, evaluate, application")
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-pos_classifier', dest='pos_classifier', required=False, default="False",
                        help='Should the system also perform POS tagging? Available only with classification.')
    parser.add_argument('-pos_tagset', dest='pos_tagset', required=False, default="coarsegrained",
                        help='Whether the POS tags should be converted. Options are: coarsegrained, finegrained')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved.')
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
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default=100000,
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
    embeddings2_dim = int(args.embeddings2_dim)
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
        train_data, lemma2id, known_lemmas, pos_types, synset2id = \
            read_data.read_data_naf(train_data_path, lemma2synsets, mode=mode, wsd_method=wsd_method,
                                    pos_tagset=pos_tagset)
    elif train_data_format == "uef":
        train_data, lemma2id, known_lemmas, pos_types, synset2id = \
            read_data.read_data_uef(train_data_path, sensekey2synset, lemma2synsets, mode=mode, wsd_method=wsd_method)
    if test_data_format == "naf":
        test_data, _, _, _, _ = \
            read_data.read_data_naf(test_data_path, lemma2synsets,  lemma2id=lemma2id, known_lemmas=known_lemmas,
                                    synset2id=synset2id, mode=mode, wsd_method=wsd_method, pos_tagset=pos_tagset)
    elif test_data_format == "uef":
        test_data, _, _, _, _ = \
            read_data.read_data_uef(test_data_path, sensekey2synset, lemma2synsets, lemma2id=lemma2id,
                                    known_lemmas=known_lemmas, synset2id=synset2id, mode=mode, wsd_method=wsd_method)

    ''' Transform the test data into the input format readable by the neural models'''
    test_inputs1, test_inputs2, test_sequence_lengths, test_labels_classif, test_labels_context, test_labels_pos, \
    test_indices, test_synsets_gold, test_pos_filters = \
        format_data.format_data(test_data, emb1_src2id, embeddings1_input, embeddings1_case, synset2id, max_seq_length,
                                embeddings1, emb2_src2id, embeddings2_input, embeddings2_case, embeddings1_dim,
                                pos_types, pos_classifier, wsd_method)

    ''' Initialize the neural model'''
    model = None
    if wsd_method == "classification":
        output_dimension = len(synset2id)
        model = architectures.classifier.ClassifierSoftmax\
            (output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
             max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
             test_sequence_lengths, test_indices, test_labels_classif, wsd_classifier, pos_classifier, len(pos_types),
             test_labels_pos)
    elif wsd_method == "context_embedding":
        output_dimension = embeddings1_dim
        model = architectures.context_embedding.ContextEmbedder\
            (output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
             max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
             test_sequence_lengths, test_indices, test_labels_context, wsd_classifier, pos_classifier, len(pos_types),
             test_labels_pos)
    elif wsd_method == "multitask":
        output_dimension = len(synset2id)
        model = architectures.multitask_wsd.MultitaskWSD\
            (output_dimension, len(embeddings1), embeddings1_dim, len(embeddings2), embeddings2_dim, batch_size,
             max_seq_length, n_hidden, n_hidden_layers, learning_rate, keep_prob, test_inputs1, test_inputs2,
             test_sequence_lengths, test_indices, test_labels_classif, test_labels_context, wsd_classifier,
             pos_classifier, len(pos_types), test_labels_pos)

    # TODO eval or train model
    ''' Run training and/or evaluation'''

    session = tf.Session()
    saver = tf.train.Saver()
    if mode == "application":
        pass
    else:
        init = tf.initialize_all_variables()
        if wsd_method == "similarity":
            feed_dict = {model.place: val_labels}
            if len(word_embeddings) > 0:
                feed_dict.update({model.emb_placeholder: word_embeddings})
            if len(lemma_embeddings) > 0:
                feed_dict.update({model.emb_placeholder_lemmas: lemma_embeddings})
            session.run(init, feed_dict=feed_dict)

        elif wsd_method == "fullsoftmax":
            feed_dict={model.emb_placeholder: word_embeddings, model.place: val_labels}
            if len(lemma_embeddings) > 0:
                session.run(init, feed_dict={model.emb_placeholder: word_embeddings, model.emb_placeholder_lemmas: lemma_embeddings,
                                             model.place: val_labels})
            else:
                session.run(init, feed_dict={model.emb_placeholder: word_embeddings, model.place: val_labels})
        elif wsd_method == "multitask":
            feed_dict = {model.place_c : val_labels[0], model.place_r : val_labels[1]}
            if len(word_embeddings) > 0:
                feed_dict.update({model.emb_placeholder: word_embeddings})
            if len(lemma_embeddings) > 0:
                feed_dict.update({model.emb_placeholder_lemmas: lemma_embeddings})
            session.run(init, feed_dict=feed_dict)

    #session.run(model.set_embeddings, feed_dict={model.emb_placeholder: word_embeddings})

    print "Start of training"
    batch_loss = 0
    best_accuracy = 0.0
    if multitask == "True":
        batch_loss_r = 0
        best_accuracy_r = 0.0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    results = open(os.path.join(args.save_path, 'results.txt'), "a", 0)
    results.write(str(args) + '\n\n')
    model_path = os.path.join(args.save_path, "model")
    for step in range(training_iters):
        offset = (step * batch_size) % (len(data) - batch_size)
        inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, \
        synsets_gold, pos_filters, pos_labels, hyp_labels, hyp_indices, freq_labels = new_batch(offset)
        if (len(labels) == 0):
            continue
        input_data = [inputs, input_lemmas, seq_lengths, labels, words_to_disambiguate, indices, pos_labels, hyp_labels,
                      hyp_indices, freq_labels]
        val_accuracy = 0.0
        if (step % 100 == 0):
            print "Step number " + str(step)
            fetches = run_epoch(session, model, input_data, keep_prob, mode="val", multitask=multitask)
            if (fetches[1] is not None):
                batch_loss += fetches[1]
            if multitask == "True" and fetches[2] is not None:
                batch_loss_r += fetches[2]
            results.write('EPOCH: %d' % step + '\n')
            results.write('Averaged minibatch loss at step ' + str(step) + ': ' + str(batch_loss/100.0) + '\n')
            if multitask == "True":
                results.write('Averaged minibatch loss (similarity) at step ' + str(step) + ': ' + str(batch_loss_r / 100.0) + '\n')
            if wsd_method == "similarity":
                val_accuracy = accuracy_cosine_distance(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold, val_pos_filters)
                results.write('Minibatch accuracy: ' + str(accuracy_cosine_distance(fetches[2], lemmas_to_disambiguate,
                                                                                    synsets_gold, pos_filters)) + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
                # Uncomment lines below in order to save the array with the modified word embeddings
                # if val_accuracy > 55.0 and val_accuracy > best_accuracy:
                #     with open(os.path.join(args.save_path, 'embeddings.pkl'), 'wb') as output:
                #                 pickle.dump(fetches[-1], output, pickle.HIGHEST_PROTOCOL)
                #     with open(os.path.join(args.save_path, 'src2id_lemmas.pkl'), 'wb') as output:
                #         pickle.dump(src2id_lemmas, output, pickle.HIGHEST_PROTOCOL)
            elif wsd_method == "fullsoftmax":
                val_accuracy, val_accuracy_pos, val_accuracy_hyp, val_accuracy_freq = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters, synset2id, val_indices, pos_classifier=pos_classifier,
                                                          wsd_classifier=wsd_classifier, logits_pos=fetches[5],
                                                          labels_pos=val_pos_labels, hypernym_classifier=hypernym_classifier,
                                                          logits_hyp=fetches[7], labels_hyp=val_labels_hyp, lemmas_hyp=val_lemmas_hyp,
                                                          pos_filters_hyp=val_pos_filters_hyp)
                results.write('Minibatch accuracy: ' + str(accuracy(fetches[2], lemmas_to_disambiguate, synsets_gold,
                                                                    pos_filters, synset2id, indices, pos_classifier=pos_classifier,
                                                                    wsd_classifier=wsd_classifier, logits_pos=fetches[4],
                                                                    labels_pos=pos_labels)[0])
                              + '\n')
                results.write('Validation accuracy: ' + str(val_accuracy) + '\n')
            elif wsd_method == "multitask":
                val_accuracy, _, _, val_accuracy_freq = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold, val_pos_filters,
                                        synset2id, synID_mapping, freq_classifier=freq_classifier, logits_freq=fetches[8], labels_freq=val_freq_labels)
                results.write('Minibatch classification accuracy: ' +
                              str(accuracy(fetches[3], lemmas_to_disambiguate, synsets_gold, pos_filters,
                                           synset2id, synID_mapping)[0]) + '\n')
                results.write('Validation classification accuracy: ' + str(val_accuracy) + '\n')
                val_accuracy_r = accuracy_cosine_distance(fetches[6], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters)
                results.write('Minibatch regression accuracy: ' +
                              str(accuracy_cosine_distance(fetches[5], lemmas_to_disambiguate, synsets_gold,
                                                           pos_filters)) + '\n')
                results.write('Validation regression accuracy: ' + str(val_accuracy_r) + '\n')

                # ops = [model.train_op, model.cost_c, model.cost_r, model.logits, model.val_logits,
                #        model.output_emb, model.val_output_emb]
            print "Validation accuracy: " + str(val_accuracy)
            if freq_classifier == "True":
                print "Validation accuracy for frequency classification is: " + str(val_accuracy_freq)
            if pos_classifier == "True":
                print "Validation accuracy for POS: " + str(val_accuracy_pos)
                results.write('Validation accuracy for POS: ' + str(val_accuracy_pos) + '\n')
            if hypernym_classifier == "True":
                print "Validation accuracy for hypernyms: " + str(val_accuracy_hyp)
                results.write('Validation accuracy for hypernyms: ' + str(val_accuracy_hyp) + '\n')
            batch_loss = 0.0
            if wsd_method == "multitask":
                batch_loss_r = 0.0
        else:
            fetches = run_epoch(session, model, input_data, keep_prob, mode="train", multitask=multitask)
            if (fetches[1] is not None):
                batch_loss += fetches[1]
            if multitask == "True" and fetches[2] is not None:
                batch_loss_r += fetches[2]

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            if pos_classifier == "True" and wsd_classifier == "True":
                val_accuracy_gold_pos, _, _, _ = accuracy(fetches[3], val_lemmas_to_disambiguate, val_synsets_gold,
                                                          val_pos_filters, synset2id, val_indices,
                                                          pos_classifier=pos_classifier, use_gold_pos="True",
                                                          logits_pos=fetches[5], labels_pos=val_pos_labels)
                results.write('Validation classification accuracy with gold POS: ' + str(val_accuracy_gold_pos) + '\n')

        if multitask == "True" and val_accuracy_r > best_accuracy_r:
            best_accurary_r = val_accuracy_r


        if (args.save_path != "None" and step == 25000 or step > 25000 and val_accuracy == best_accuracy):
            for file in os.listdir(model_path):
                os.remove(os.path.join(model_path, file))
            saver.save(session, os.path.join(args.save_path, "model/model.ckpt"), global_step=step)
            if (step == 25000):
                with open(os.path.join(args.save_path, 'lemma2synsets.pkl'), 'wb') as output:
                    pickle.dump(lemma2synsets, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'lemma2id.pkl'), 'wb') as output:
                    pickle.dump(lemma2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'synset2id.pkl'), 'wb') as output:
                    pickle.dump(synset2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2synset.pkl'), 'wb') as output:
                    pickle.dump(id2synset, output, pickle.HIGHEST_PROTOCOL)

    results.write('\n\n\n' + 'Best result is: ' + str(best_accuracy))
    if multitask == "True":
        results.write('\n\n\n' + 'Best result (similarity) is: ' + str(best_accuracy_r))
    results.close()

    print "This is the end."
