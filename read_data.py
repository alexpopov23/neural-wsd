import collections
import os

import _elementtree

def get_wordnet_lexicon (lexicon_path):
    """Reads the WordNet dictionary

    Args:
        lexicon_path: A string, the path to the dictionary

    Returns:
        lemma2synsets: A dictionary, maps lemmas to synset IDs
        lemma2id: A dictionary, maps lemmas to integer IDs
        synset2id: A dictionary, maps synset IDs to integer IDs
    """
    lemma2synsets = {}
    lemma2id = {}
    synset2id = {}
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
    index_l, index_s = 0, 0
    for lemma, synsets in lemma2synsets.iteritems():
        lemma2id[lemma] = index_l
        index_l += 1
        for synset in synsets:
            if synset not in synset2id:
                synset2id[synset] = index_s
                index_s += 1
    return lemma2synsets, lemma2id, synset2id

def read_naf_file (path):
    """Reads file in NAF format

    Args:
        path: A string, the path to the NAF file

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
    for wf in wfs:
        wf_id = int(wf.get("id")[1:])
        wf_text = wf.text
        wf_sent = wf.get("sent")
        corpus[wf_id] = [wf_sent, wf_text]
    terms = doc.find("terms")
    for term in terms.findall("term"):
        lemma = term.get("lemma")
        pos = term.get("pos")
        id = int(term.find("span").find("target").get("id")[1:])
        synset = "unspecified"
        extRefs = term.find("externalReferences")
        if extRefs != None:
            for extRef in extRefs.findall("externalRef"):
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

def read_naf_folder (path, known_lemmas=set(), synset2id={}, mode="train", lexicon_path=None):
    """Reads folders with files in NAF format

    Args:
        path: A string, the path to the data folder
        lemma2synsets:  A dictionary, mapping observed lemmas to synset IDs (empty when reading training data)
        lemma2id: A dictionary, mapping observed lemmas to integer IDs (empty when reading training data)
        synset2id: A dictionary, mapping synsets to integer IDs (empty when reading training data)
        mode: A string, indicates whether the data is for training or testing

    Returns:
        data: A list of lists; each sentence contains "words" represented
              in the format: [wordform, lemma, POS, [synset1, ..., synsetN]]
        lemma2synsets: A dictionary, mapping observed lemmas to synset IDs
        lemma2id: A dictionary, mapping observed lemmas to integer IDs
        known_lemmas: A set, all lemmas seen in training
        synset2id: A dictionary, mapping synsets to integer IDs
    """
    data = []
    for f in os.listdir(path):
        new_data, new_lemmas = read_naf_file(os.path.join(path, f))
        known_lemmas.update(new_lemmas)
        data.extend(new_data)
    for sentence in data:
        for word in sentence:
            synset = word[3]
            if synset != "unspecified":
                word.extend([synset2id[synset]])
            else:
                word.extend([-1])
    return data, known_lemmas

def read_data_uniroma (path, sensekey2synset, lemma2synsets={}, lemma2id={}, known_lemmas=set(), synset2id={}, synID_mapping={},
                       wsd_method="classification", mode="train", f_lex=None):

    data = []
    if mode == "train":
        # get lexicon from the WordNet files
        lexicon = open(f_lex, "r")
        lines = lexicon.readlines()
        for line in lines:
            fields = line.split(" ")
            lemma, synsets = fields[0], fields[1:]
            most_freq = -1
            for entry in synsets:
                synset = entry[:10].strip()
                if lemma not in lemma2synsets:
                    lemma2synsets[lemma] = [synset]
                else:
                    lemma2synsets[lemma].append(synset)
    #sensekey2synset = get_sensekey2synset()
    #sensekey2synset = pickle.load(open("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/sensekey2synset.pkl", "rb"))
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
        entries = line.strip().split()
        code = entries[0]
        keys = entries[1:]
        codes2keys[code] = keys
    #with open(os.path.join(path, path_data)) as f:
    #    xml = f.read()
    #tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")
    tree = ET.parse(os.path.join(path, path_data))
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
                    if mode == "train":
                        known_lemmas.add(lemma)
                    pos = element.get("pos")
                    if element.tag == "instance":
                        synsets = [sensekey2synset[key] for key in codes2keys[element.get("id")]]
                        # TODO: fix in the generation of the dictionary, this here is a needless check
                        for synset in synsets:
                            if synset.endswith("-s"):
                                synsets[synsets.index(synset)] = synset.replace("-s", "-a")
                    else:
                        synsets = ["unspecified"]
                    current_sentence.append([wordform, lemma, pos, synsets])
                data.append(current_sentence)
    if mode == "train":
        lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
        index_l = 0
        index_s = 0
        index_s_map = 0
        if wsd_method == "fullsoftmax" or wsd_method == "multitask":
            synset2id['notseen-n'], synset2id['notseen-v'], synset2id['notseen-a'], synset2id['notseen-r'] = 0, 1, 2, 3
            if wsd_method == "multitask":
                synID_mapping.update({0:0, 1:1, 2:2, 3:3})
            index_s = 4
            index_s_map = 4
        for lemma, synsets in lemma2synsets.iteritems():
            if wsd_method == "fullsoftmax" and lemma not in known_lemmas:
                continue
            lemma2id[lemma] = index_l
            index_l += 1
            for synset in synsets:
                if synset not in synset2id:
                    synset2id[synset] = index_s
                    index_s += 1
                if wsd_method == "multitask" and lemma in known_lemmas:
                    indx_to_map = synset2id[synset]
                    if indx_to_map in synID_mapping:
                        continue
                    synID_mapping[indx_to_map] = index_s_map
                    index_s_map += 1

    words_to_disambiguate = []
    count_ambig = 0
    count_missing1 = 0
    count_missing2 = 0
    count_inst = 0
    for i, sentence in enumerate(data):
        for word in sentence:
            if word[-1][0] != "unspecified":
                count_inst += 1
                if len(word[-1]) > 1:
                    count_ambig += 1
                synsets = []
                # check if lemma is known
                if (wsd_method == "fullsoftmax" or wsd_method == "multitask") and word[1] not in known_lemmas:
                    if len(lemma2synsets[word[1]]) == 1:
                        count_missing1 += 1
                    elif len(lemma2synsets[word[1]]) > 1:
                        count_missing2 += 1
                    if word[2] == "NOUN":
                        synsets.append(synset2id['notseen-n'])
                    elif word[2] == "VERB":
                        synsets.append(synset2id['notseen-v'])
                    elif word[2] == "ADJ":
                        synsets.append(synset2id['notseen-a'])
                    elif word[2] == "ADV":
                        synsets.append(synset2id['notseen-r'])
                    #lemma2synsets[word[1]] = [syn]
                # check if synset is known
                else:
                    for syn in word[-1]:
                        synsets.append(synset2id[syn])
                word.append(synsets)
                words_to_disambiguate.append(word)
            else:
                word.append([-1])
    return data, lemma2synsets, lemma2id, synset2id, synID_mapping, known_lemmas