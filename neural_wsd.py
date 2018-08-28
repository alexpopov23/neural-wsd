import argparse
import pickle

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
    parser.add_argument('-embeddings2_dim', dest='embeddings2_dim', required=False, default=300,
                        help='Size of the additional embeddings.')
    # parser.add_argument('-embeddings1_format', dest='embeddings1_format', required=False, default="gensim",
    #                     help='Which method is used for loading the pretrained embeddings? Options: gensim, glove?')
    # parser.add_argument('-embeddings2_format', dest='embeddings2_format', required=False, default="gensim",
    #                     help='Which method is used for loading the additional pretrained embeddings? Options: gensim, glove?')
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
    # parser.add_argument('-synset_mapping', dest='synset_mapping', required=False,
    #                     help='A mapping between the synset embedding IDs and WordNet, if such is necessary.')
    # parser.add_argument('-lexicon_mode', dest='lexicon_mode', required=False, default="full_dictionary",
    #                     help='Whether to use a lexicon or only the senses attested in the corpora: *full_dictionary* or *attested_senses*.')


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
        embeddings2, emb2_src2id, emb2_src2id = None, None, None

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
    if wsd_method == "classifier":
        model = architectures.classifier.ClassifierSoftmax()

    if wsd_method == "similarity":
        model = ModelVectorSimilarity(word_embedding_input, output_embedding_dim, lemma_embedding_dim, vocab_size_lemmas,
                                      batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths,
                                      val_words_to_disambiguate, val_indices, val_labels, word_embedding_dim, vocab_size)
    elif wsd_method == "fullsoftmax":
        model = ModelSingleSoftmax(synset2id, word_embedding_dim, vocab_size, batch_size, seq_width, n_hidden,
                                   n_hidden_layers, val_inputs, val_input_lemmas, val_seq_lengths, val_words_to_disambiguate,
                                   val_indices, val_labels, lemma_embedding_dim, len(src2id_lemmas))
    elif wsd_method == "multitask":
        if word_embedding_input == "wordform":
            output_embedding_dim = word_embedding_dim
        else:
            output_embedding_dim = lemma_embedding_dim
        model = ModelMultiTaskLearning(word_embedding_input, synID_mapping, output_embedding_dim, lemma_embedding_dim,
                                       vocab_size_lemmas, batch_size, seq_width, n_hidden, val_inputs, val_seq_lengths,
                                       val_words_to_disambiguate, val_indices, val_labels[0], val_labels[1],
                                       word_embedding_dim, vocab_size)

    # TODO initialize model
    # TODO eval or train model

    print "This is the end."
