import numpy

from sklearn.metrics.pairwise import cosine_similarity


def accuracy_classification(logits, target_lemmas, synsets_gold, pos_filters, synset2id, lemma2synsets, known_lemmas,
                            wsd_classifier=True, pos_classifier=False, logits_pos=None, pos_gold=None):
    """Calculates the classification accuracy of a model run - can be used to calculate WSD and POS tagging accuracy

    Args:
        logits: A list of arrays, each holding a probability distribution
        target_lemmas: A list of strings, the lemmas of the words to be disambiguated, in order of their appearance
        synsets_gold: A list of strings, the correct synsets according to the gold data, in order of appearance
        pos_filters: A list of strings, the POS tags associated with the words to be disambiguated, used for filtering
                     out irrelevant synsets from the evaluation
        synset2id: A dictionary, mapping the possible synsets (of lemmas seen in training) to their position in the
                   vocabulary (and in the softmax output vector)
        lemma2synsets: A dictionary, mapping all lemmas in WordNet to their corresponding synsets
        known_lemmas: A set, giving all lemmas seen in training
        wsd_classifier: A boolean, indicating whether WSD accuracy should be computed
        pos_classifier: A boolean, indicating whether POS tagging accuracy should be computed
        logits_pos: A list of arrays, each holding a probability distribution
        pos_gold: A list of arrays, each being a one-hot representation of a gold POS label per word

    Returns:
        accuracy_wsd: A float, the accuracy for the WSD task
        accuracy_pos: A float, the accuracy for the POS tagging task
        book_keeping: A list, holds the number of matching and evaluation cases

    """
    accuracy_wsd, matching_cases, eval_cases = 0.0, 0, 0
    if wsd_classifier is True:
        for i, logit in enumerate(logits):
            max, max_id = -10000, -1
            gold_tags = synsets_gold[i]
            pos = pos_filters[i]
            lemma = target_lemmas[i]
            if lemma not in known_lemmas:
                for synset in lemma2synsets[lemma]:
                    # make sure we only evaluate on synsets of the correct POS category
                    if pos != synset.split("-")[1]:
                        continue
                    else:
                        max_id = synset
                        break
            else:
                for synset in lemma2synsets[lemma]:
                    id = synset2id[synset]
                    # make sure we only evaluate on synsets of the correct POS category
                    if pos != synset.split("-")[1]:
                        continue
                    if logit[id] > max:
                        max = logit[id]
                        max_id = synset
            if max_id in gold_tags:
                matching_cases += 1
            eval_cases += 1
        accuracy_wsd = (100.0 * matching_cases) / eval_cases
    accuracy_pos, matching_cases_pos, eval_cases_pos = 0.0, 0, 0
    if pos_classifier is True:
        for i, logit_pos in enumerate(logits_pos):
            if numpy.amax(pos_gold[i]) == 0:
                continue
            if numpy.argmax(logit_pos) == numpy.argmax(pos_gold[i]):
                matching_cases_pos += 1
            eval_cases_pos += 1
        accuracy_pos = (100.0 * matching_cases_pos) / eval_cases_pos
    book_keeping = [matching_cases, eval_cases, matching_cases_pos, eval_cases_pos]
    return accuracy_wsd, accuracy_pos, book_keeping


def accuracy_cosine_distance(context_embeddings, target_lemmas, synsets_gold, pos_filters, lemma2synsets,
                             sense_embeddings, embeddings_src2id):
    """Calculates the accuracy of a model run, using cosine similarity to find the admissible synset embedding which is
    closest to the context embedding produced by the network

    Args:
        context_embeddings: A list of arrays, each holding a context embedding in the same VSM as used for the input
        target_lemmas: A list of strings, the lemmas of the words to be disambiguated, in order of their appearance
        synsets_gold: A list of strings, the correct synsets according to the gold data, in order of appearance
        pos_filters: A list of strings, the POS tags associated with the words to be disambiguated, used for filtering
                     out irrelevant synsets from the evaluation
        lemma2synsets: A dictionary, mapping all lemmas in WordNet to their corresponding synsets
        sense_embeddings: A list of arrays, each one being a word or concept/synset representation in the VSM
        embeddings_src2id: A dictionary, mapping strings (synset IDs) to integer indices into the list of embeddings

    Returns:
        accuracy: A float, the accuracy for the WSD task
        book_keeping: A tuple, holds the number of matching and evaluation cases

    """
    matching_cases, eval_cases = 0, 0
    for i, context_embedding in enumerate(context_embeddings):
        lemma = target_lemmas[i]
        possible_synsets = lemma2synsets[lemma]
        best_fit = "None"
        max_similarity = -10000.0
        pos = pos_filters[i]
        for j, synset in enumerate(possible_synsets):
            if pos != synset.split("-")[1]:
                continue
            if synset not in embeddings_src2id:
                if max_similarity == -10000:
                    best_fit = synset
                continue
            syn_id = embeddings_src2id[synset]
            cos_sim = cosine_similarity(context_embedding.reshape(1, -1), sense_embeddings[syn_id].reshape(1, -1))[0][0]
            if cos_sim > max_similarity:
                max_similarity = cos_sim
                best_fit = synset
        if best_fit in synsets_gold[i]:
            matching_cases += 1
        eval_cases += 1
    accuracy = (100.0 * matching_cases) / eval_cases
    book_keeping = (matching_cases, eval_cases)
    return accuracy, book_keeping