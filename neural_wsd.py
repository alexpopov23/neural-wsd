import argparse
import os
import pickle

import load_embeddings
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
    parser.add_argument("-mode", dest="mode", required=False, default="train",
                        help="Is this is a training, evaluation or application run? Options: train, evaluate, application")
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved.')
    parser.add_argument('-sensekey2synset', dest='sensekey2synset', required=False,
                        help='Path to mapping between sense annotations in the corpus and synset IDs in WordNet.')
    parser.add_argument('-sequence_width', dest='sequence_width', required=False, default=63,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-test_data', dest='test_data', required=False,
                        help='The path to the gold corpus used for testing.')
    parser.add_argument("-test_data_format", dest="test_data_format", required=False,
                        help="Specifies the format of the evaluation corpus. Options: naf, uniroma")
    parser.add_argument('-train_data', dest='train_data', required=False,
                        help='The path to the gold corpus used for training.')
    parser.add_argument("-train_data_format", dest="train_data_format", required=False,
                        help="Specifies the format of the training corpus. Options: naf, uniroma")
    parser.add_argument('-training_iterations', dest='training_iterations', required=False, default=100000,
                        help='How many iterations the network should train for.')
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
    embeddings1_dim = args.embeddings1_dim
    # embeddings1_format = args.embeddings1_format
    embeddings1_input = args.embeddings1_input
    embeddings2_path = args.embeddings2_path
    if embeddings2_path is not None:
        embeddings2_case = args.embeddings2_case
        embeddings2_dim = args.embeddings2_dim
        # embeddings2_format = args.embeddings2_format
        embeddings2_input = args.embeddings2_input
    keep_prob = float(args.keep_prob)
    learning_rate = float(args.learning_rate)
    lexicon_path = args.lexicon_path
    mode = args.mode
    n_hidden = int(args.n_hidden)
    n_hidden_layers = int(args.n_hidden_layers)
    save_path = args.save_path
    sensekey2synset_path = args.sensekey2synset_path
    sequence_width = args.sequence_width
    test_data = args.test_data
    test_data_format = args.test_data_format
    train_data = args.train_data
    train_data_format = args.train_data_format
    training_iterations = args.training_iterations
    wsd_method = args.wsd_method

    ''' Load the embedding model(s) that will be used '''
    embeddings1, emb1_src2id, emb1_id2src = load_embeddings.load(embeddings1_path)
    if embeddings2_path is not None:
        embeddings2, emb2_src2id, emb2_id2src = load_embeddings.load(embeddings2_path)

    # TODO load training/test data and auxiliary resources
    # Obtain mapping between data-specific sense codes and WN synset IDs
    if train_data_format == "uniroma" or test_data_format == "uniroma":
        sensekey2synset = pickle.load(open(sensekey2synset_path, "rb"))
    lemma2synsets, lemma2id, synset2id = read_data.get_wordnet_lexicon(lexicon_path)
    # Read train and test data files into a standard format+
    if train_data_format == "naf":
        pass
    # if data_source == "naf":
    #     data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos = \
    #         data_ops.read_folder_semcor(data, lexicon_mode=lexicon_mode, f_lex=lexicon)
    # elif data_source == "uniroma":
    #     data, lemma2synsets, lemma2id, synset2id, synID_mapping, id2synset, id2pos, known_lemmas, synset2freq = \
    #         data_ops.read_data_uniroma(data, sensekey2synset, wsd_method=wsd_method, f_lex=lexicon)


    # TODO format test data
    # TODO initialize model
    # TODO eval or train model

    print "This is the end."
