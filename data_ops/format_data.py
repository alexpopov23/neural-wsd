import copy

import numpy

import globals

def get_embedding_id(word, input, case, src2id):
    """Takes data about a word and obtains the relevant embedding id

    Args:
        word: A list ([wordform, lemma])
        input: A string; either "wordform" or "lemma"
        case: A string; either "lowercase" or "mixedcase"
        src2id: A dictionary, the mapping between strings and embedding IDs

    Returns:
        embedding_id: An integer, the index into the embeddings model

    """
    if input == "wordform":
        if word[0].lower() not in src2id:
            return src2id["UNK"]
        if case == "lowercase":
            embedding_id = src2id[word[0].lower()]
        elif case == "mixedcase":
            embedding_id = src2id[word[0]]
    elif input == "lemma":
        if word[1] not in src2id:
            return src2id["UNK"]
        embedding_id = src2id[word[1]]
    return embedding_id


def format_data(data, emb1_src2id, emb1_input, emb1_case, synset2id, max_seq_length, embeddings1=None,
                 emb2_src2id=None, emb2_input=None, emb2_case=None, emb_dim=None,
                 pos_types=None, pos_classifier=False, wsd_method="classification"):
    """Takes a training/test corpus and transforms it to be readable by the neural models

    Args:
        data: A list of lists, stores information about words which are grouped into sentences
        emb1_src2id: A dictionary, maps strings to embedding integer IDs
        emb1_input: A string, indicates whether the embeddings apply to wordforms or lemmas
        emb1_case: A string, indicates the case of the embedding strings
        synset2id: A dictionary, mapping synsets to integer IDs
        max_seq_length: An integer, the maximum allowed length per sentence
        embeddings1: An array, the primary embeddings (necessary for preparing the context embedding data)
        emb2_src2id: A dictionary, maps strings to embedding integer IDs
        emb2_input: A string, indicates whether the embeddings apply to wordforms or lemmas
        emb2_case: A string, indicates the case of the embedding strings
        emb_dim: An integer, indicates the size of the embeddings; needed by the context embedding method
        pos_types: A dictionary, all POS tags seen in training and their mappings to integer IDs
        pos_classifier: A boolean, indicates whether POS labels are needed by the system
        wsd_method: A string, indicates the disamguation method used ("classification", "context_embedding", "multitask")

    Returns:
        inputs1: A list of lists, the integer IDs for the inputs in the primary embedding model
        inputs2: A list of lists, the integer IDs for the inputs in the auxiliary embedding model, if in use
        sequence_lengths: A list of ints, the lengths of the individual sentences
        labels_classif: A list of (one-hot) arrays, the gold labels for the WSD classification method, if in use
        labels_context: A list of arrays, the "gold" embeddings for the context embedding WSD method, if in use
        labels_pos: A list of (one-hot) arrays, the gold labels for the POS classification method, if in use
        indices: A list of integers, indexes which words in the data are to be disambiguated
        synsets_gold: A list of strings, provides the gold synset IDs
        pos_filters: A list of strings, provides the POS tags per word (simple tagset: n, v, a, r)

    """
    inputs1, inputs2, sequence_lengths, labels_classif, labels_context, labels_pos, indices, target_lemmas, \
    synsets_gold, pos_filters = [], [], [], [], [], [], [], [], [], []
    zero_pos_label = numpy.zeros(len(pos_types), dtype=int)
    counter = 0
    for i, sentence in enumerate(data):
        if len(sentence) > max_seq_length:
            sentence = sentence[:max_seq_length]
        # Use the following lists to store formatted data for the current sentence
        c_input1, c_input2, c_labels_classif, c_labels_context, c_labels_pos, c_synsets, c_pos_filters = \
            [], [], [], [], [], [], []
        for j, word in enumerate(sentence):
            # Obtain the embedding IDs per word
            c_input1.append(get_embedding_id(word, emb1_input, emb1_case, emb1_src2id))
            if emb2_src2id is not None:
                c_input2.append(get_embedding_id(word, emb2_input, emb2_case, emb2_src2id))
            # Obtain the synset gold labels / embeddings
            if (word[4][0] > -1):
                if wsd_method == "classification" or wsd_method == "multitask":
                    c_label_classif = numpy.zeros(len(synset2id), dtype=numpy.float32)
                    for synset_id in word[4]:
                        if synset_id < len(synset2id):
                            c_label_classif[synset_id] = 1.0/len(word[4])
                        else:
                            if word[2] in globals.pos_map:
                                pos = globals.pos_map[word[2]]
                            else:
                                pos = word[2]
                            if pos == "NOUN":
                                c_label_classif[synset2id['notseen-n']] = 1.0 / len(word[4])
                            elif pos == "VERB":
                                c_label_classif[synset2id['notseen-v']] = 1.0 / len(word[4])
                            elif pos == "ADJ":
                                c_label_classif[synset2id['notseen-a']] = 1.0 / len(word[4])
                            elif pos == "ADV":
                                c_label_classif[synset2id['notseen-r']] = 1.0 / len(word[4])
                    c_labels_classif.append(c_label_classif)
                if wsd_method == "context_embedding" or wsd_method == "multitask":
                    for synset in word[3]:
                        c_label_context = numpy.zeros([emb_dim], dtype=numpy.float32)
                        if synset in emb1_src2id:
                            c_label_context += embeddings1[emb1_src2id[synset]]
                    c_label_context = c_label_context / len(word[4])
                    c_labels_context.append(c_label_context)
                c_synsets.append(word[3])
                target_lemmas.append(word[1])
                if word[2] in globals.pos_map_simple:
                    c_pos_filters.append(globals.pos_map_simple[word[2]])
                else:
                    c_pos_filters.append(globals.pos_map[word[2]])
                indices.append(counter)
            if pos_classifier is True:
                c_pos_label = copy.copy(zero_pos_label)
                c_pos_label[pos_types[word[2]]] = 1
                c_labels_pos.append(c_pos_label)
            counter += 1
        sequence_lengths.append(len(c_input1))
        padding_size = max_seq_length - len(c_input1)
        counter += padding_size
        c_input1 += padding_size * [emb1_src2id["UNK"]]
        c_input1 = numpy.asarray(c_input1)
        inputs1.append(c_input1)
        if emb2_src2id is not None:
            c_input2 += padding_size * [emb2_src2id["UNK"]]
            c_input2 = numpy.asarray(c_input2)
            inputs2.append(c_input2)
        if pos_classifier is True:
            c_labels_pos += padding_size * [zero_pos_label]
            labels_pos.extend(c_labels_pos)
        if wsd_method == "classification" or wsd_method == "multitask":
            labels_classif.extend(c_labels_classif)
        if wsd_method == "context_embedding" or wsd_method == "multitask":
            labels_context.extend(c_labels_context)
        synsets_gold.extend(c_synsets)
        pos_filters.extend(c_pos_filters)
    inputs1 = numpy.asarray(inputs1)
    inputs2 = numpy.asarray(inputs2)
    sequence_lengths = numpy.asarray(sequence_lengths)
    labels_classif = numpy.asarray(labels_classif)
    labels_context = numpy.asarray(labels_context)
    labels_pos = numpy.asarray(labels_pos)
    indices = numpy.asarray(indices)
    return inputs1, inputs2, sequence_lengths, labels_classif, labels_context, labels_pos, indices, target_lemmas,\
           synsets_gold, pos_filters


def new_batch(offset, batch_size, data, emb1_src2id, embeddings1_input, embeddings1_case, synset2id, max_seq_length,
              embeddings1, emb2_src2id, embeddings2_input, embeddings2_case, embeddings1_dim, pos_types, pos_classifier,
              wsd_method):
    """Create a new batch from the training data. See format_data() for most arguments

    Additional args:
        offset: An int, the position where the new batch should be extracted from the training data
        batch_size: An int, the size of the new batch

    Returns:
        see format_data()

    """
    batch = data[offset:(offset + batch_size)]
    inputs1, inputs2, sequence_lengths, labels_classif, labels_context, labels_pos, indices, target_lemmas, \
    synsets_gold, pos_filters = \
        format_data(batch, emb1_src2id, embeddings1_input, embeddings1_case, synset2id,
                    max_seq_length, embeddings1, emb2_src2id, embeddings2_input, embeddings2_case, embeddings1_dim,
                    pos_types, pos_classifier, wsd_method)
    return inputs1, inputs2, sequence_lengths, labels_classif, labels_context, labels_pos, indices, target_lemmas,\
           synsets_gold, pos_filters