import collections
import os

import _elementtree

import globals


def get_wordnet_lexicon(lexicon_path):
    """Reads the WordNet dictionary

    Args:
        lexicon_path: A string, the path to the dictionary

    Returns:
        lemma2synsets: A dictionary, maps lemmas to synset IDs

    """
    lemma2synsets = {}
    lexicon = open(lexicon_path, "r")
    for line in lexicon.readlines():
        fields = line.split(" ")
        lemma, synsets = fields[0], fields[1:]
        for entry in synsets:
            synset = entry[:10].strip()
            if lemma not in lemma2synsets:
                lemma2synsets[lemma] = [synset]
            else:
                lemma2synsets[lemma].append(synset)
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    return lemma2synsets


def get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id, synset2id):
    """Constructs mappings between lemmas and integer IDs, synsets and integerIDs

    Args:
        wsd_method: A string ("classification"|"context_embedding"|"multitask")
        lemma2synsets: A dictionary, maps lemmas to synset IDs
        known_lemmas: A set of lemmas seen in the training data
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty)
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs (empty)

    Returns:
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs

    """
    index_l, index_s = 0, 0
    if wsd_method == "classification" or wsd_method == "multitask":
        synset2id['notseen-n'], synset2id['notseen-v'], synset2id['notseen-a'], synset2id['notseen-r'] = 0, 1, 2, 3
        index_s = 4
    for lemma, synsets in lemma2synsets.iteritems():
        if (wsd_method == "classification" or wsd_method == "multitask") and lemma not in known_lemmas:
            continue
        lemma2id[lemma] = index_l
        index_l += 1
        for synset in synsets:
            if synset not in synset2id:
                synset2id[synset] = index_s
                index_s += 1
    return lemma2id, synset2id


def add_synset_ids(wsd_method, data, known_lemmas, synset2id):
    """Adds integer IDs for the synset annotations of words in data

    Args:
        wsd_method: A string ("classification"|"context_embedding"|"multitask")
        data:   A list of lists; each sentence contains "words" represented
                in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        known_lemmas: A set of lemmas seen in the training data
        synset2id: A dictionary, mapping synsets to integer IDs

    Returns:
        data:   A list of lists; each sentence contains "words" represented
                in the format: [wordform, lemma, POS, [synset1, ..., synsetN], [synsetID1, ..., synsetIDN]]

    """
    for sentence in data:
        for word in sentence:
            synsets = word[3]
            if synsets[0] != "unspecified":
                synset_ids = []
                lemma = word[1]
                pos = word[2]
                if (wsd_method == "classification" or wsd_method == "multitask") and lemma not in known_lemmas:
                    if pos == "NOUN" or pos in globals.pos_map and globals.pos_map[pos] == "NOUN":
                        synset_ids.append(synset2id['notseen-n'])
                    elif pos == "VERB" or pos in globals.pos_map and globals.pos_map[pos] == "VERB":
                        synset_ids.append(synset2id['notseen-v'])
                    elif pos == "ADJ" or pos in globals.pos_map and globals.pos_map[pos] == "ADJ":
                        synset_ids.append(synset2id['notseen-a'])
                    elif pos == "ADV" or pos in globals.pos_map and globals.pos_map[pos] == "ADV":
                        synset_ids.append(synset2id['notseen-r'])
                else:
                    for synset in synsets:
                        synset_ids.append(synset2id[synset])
                word.append(synset_ids)
            else:
                word.append([-1])
    return data


def read_naf_file(path, pos_tagset, pos_types):
    """Reads file in NAF format

    Args:
        path: A string, the path to the NAF file
        pos_tagset: A string, indicates whether POS tags should be coarse- or fine-grained
        pos_types: A dictionary, maps known POS tags to unique integer IDs

    Returns:
        sentences: A list of lists; each sentence contains "words" represented
                   in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]

    """
    tree = _elementtree.parse(path)
    doc = tree.getroot()
    text = doc.find("text")
    wfs = text.findall("wf")
    corpus = {}
    known_lemmas = set()
    pos_count = len(pos_types)
    for wf in wfs:
        wf_id = int(wf.get("id")[1:])
        wf_text = wf.text
        wf_sent = wf.get("sent")
        corpus[wf_id] = [wf_sent, wf_text]
    terms = doc.find("terms")
    for term in terms.findall("term"):
        lemma = term.get("lemma")
        pos = term.get("pos")
        if pos in globals.pos_normalize:
            pos = globals.pos_normalize[pos]
        if pos_tagset == "coarsegrained":
            if pos in globals.pos_map:
                pos = globals.pos_map[pos]
        if pos not in pos_types:
            pos_types[pos] = pos_count
            pos_count += 1
        id = int(term.find("span").find("target").get("id")[1:])
        synset = "unspecified"
        ext_refs = term.find("externalReferences")
        if ext_refs is not None:
            for extRef in ext_refs.findall("externalRef"):
                resource = extRef.get("resource")
                if resource == "WordNet-eng30" or resource == "WordNet-3.0":
                    reftype = extRef.get("reftype")
                    if reftype == "synset" or reftype == "ilidef":
                        synset = extRef.get("reference")[-10:]
                        if lemma not in known_lemmas:
                            known_lemmas.add(lemma)
        corpus[id].extend([lemma, pos, [synset]])
    corpus = collections.OrderedDict(sorted(corpus.items()))
    sentences = []
    current_sentence = []
    sent_counter = 1
    for word in corpus.iterkeys():
        if len(corpus[word]) == 2:
            lemma = corpus[word][1]
            corpus[word].extend([lemma, ".", ["unspecified"]])
        if int(corpus[word][0]) == sent_counter:
            current_sentence.append(corpus[word][1:])
        else:
            if sent_counter != 0:
                sentences.append(current_sentence)
            sent_counter += 1
            current_sentence = []
            current_sentence.append(corpus[word][1:])
    sentences.append(current_sentence)
    return sentences, known_lemmas


def read_data_naf(path, lemma2synsets, lemma2id={}, known_lemmas=set(), synset2id={}, for_training=True,
                  wsd_method="classification", pos_tagset="coarsegrained"):
    """Reads folders with files in NAF format

    Args:
        path: A string, the path to the data folder
        lemma2synsets: A dictionary, mapping lemmas to lists of synset IDs
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty when reading training data)
        known_lemmas: A set of lemmas seen in the training data (empty when reading training data)
        synset2id: A dictionary, mapping synsets to integer IDs (empty when reading training data)
        for_training: A boolean, indicates whether the data is for training or testing
        wsd_method: A string, indicates the disamguation method used ("classification", "context_embedding", "multitask")
        pos_tagset: A string, indicates whether POS tags should be coarse- or fine-grained

    Returns:
        data: A list of lists; each sentence contains "words" represented
              in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set, all lemmas seen in training
        pos_types: A dictionary, all POS tags seen in training and their mappings to integer IDs
        synset2id: A dictionary, mapping synsets to integer IDs

    """
    data = []
    pos_types = {}
    for f in os.listdir(path):
        new_data, new_lemmas = read_naf_file(os.path.join(path, f), pos_tagset, pos_types)
        known_lemmas.update(new_lemmas)
        data.extend(new_data)
    pos_types["."] = len(pos_types)
    if for_training is True:
        lemma2id, synset2id = get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id,
                                                                    synset2id)
    data = add_synset_ids(wsd_method, data, known_lemmas, synset2id)
    return data, lemma2id, known_lemmas, pos_types, synset2id


def read_data_uef(path, sensekey2synset, lemma2synsets, lemma2id={}, known_lemmas=set(), synset2id={},
                  for_training=True, wsd_method="classification"):
    """Reads a corpus in the Universal Evaluation Framework (UEF) format

    Args:
        path: A string, the path to the data folder
        sensekey2synset: A dictionary, mapping sense IDs to synset IDs
        lemma2synsets: A dictionary, mapping lemmas to lists of synset IDs
        lemma2id: A dictionary, mapping lemmas to integer IDs (empty when reading training data)
        known_lemmas: A set of lemmas seen in the training data (empty when reading training data)
        synset2id: A dictionary, mapping synsets to integer IDs (empty when reading training data)
        for_training: A boolean, indicates whether the data is for training or testing
        wsd_method: A string, indicates the disamguation method used ("classification", "context_embedding", "multitask")

    Returns:
        data: A list of lists; each sentence contains "words" represented
              in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        lemma2id: A dictionary, mapping lemmas to integer IDs
        known_lemmas: A set, all lemmas seen in training
        pos_types: A dictionary, all POS tags seen in training and their mappings to integer IDs
        synset2id: A dictionary, mapping synsets to integer IDs

    """
    data = []
    pos_types, pos_count = {}, 0
    path_data = ""
    path_keys = ""
    for f in os.listdir(path):
        if f.endswith(".xml"):
            path_data = f
        elif f.endswith(".txt"):
            path_keys = f
    codes2keys = {}
    f_codes2keys = open(os.path.join(path, path_keys), "r")
    for line in f_codes2keys.readlines():
        fields = line.strip().split()
        code = fields[0]
        keys = fields[1:]
        codes2keys[code] = keys
    tree = _elementtree.parse(os.path.join(path, path_data))
    doc = tree.getroot()
    corpora = doc.findall("corpus")
    for corpus in corpora:
        texts = corpus.findall("text")
        for text in texts:
            sentences = text.findall("sentence")
            for sentence in sentences:
                current_sentence = []
                elements = sentence.findall(".//")
                for element in elements:
                    wordform = element.text
                    lemma = element.get("lemma")
                    if for_training is True:
                        known_lemmas.add(lemma)
                    pos = element.get("pos")
                    if pos not in pos_types:
                        pos_types[pos] = pos_count
                        pos_count += 1
                    if element.tag == "instance":
                        synsets = [sensekey2synset[key] for key in codes2keys[element.get("id")]]
                    else:
                        synsets = ["unspecified"]
                    current_sentence.append([wordform, lemma, pos, synsets])
                data.append(current_sentence)
    if for_training is True:
        lemma2id, synset2id = get_lemma_synset_maps(wsd_method, lemma2synsets, known_lemmas, lemma2id,
                                                                    synset2id)
    data = add_synset_ids(wsd_method, data, known_lemmas, synset2id)
    return data, lemma2id, known_lemmas, pos_types, synset2id
