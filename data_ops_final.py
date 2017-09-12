import collections
import os
import random
import pickle

import numpy as np
import _elementtree as ET


from copy import copy
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
from nltk.corpus import wordnet

npa = np.array

# read a single NAF-style Semcor file
def read_file_semcor (path, mode="full_dictionary"):
    if mode == "full_dictionary":
        dictionary = set()
    elif mode == "attested_senses":
        dictionary = {}
    tree = ET.parse(path)
    doc = tree.getroot()
    text = doc.find("text")
    wfs = text.findall("wf")
    corpus = {}
    for wf in wfs:
        wf_id = int(wf.get("id")[1:])
        wf_text = wf.text
        wf_sent = wf.get("sent")
        corpus[wf_id] = [wf_sent, wf_text]
    terms = doc.find("terms")
    for term in terms.findall("term"):
        lemma = term.get("lemma")
        pos = term.get("pos")
        # if pos.startswith("NN"):
        #     pos = "n"
        # elif pos.startswith("VB"):
        #     pos = "v"
        # elif pos.startswith("JJ"):
        #     pos = "a"
        # elif pos.startswith("R"):
        #     pos = "r"
        id = int(term.find("span").find("target").get("id")[1:])
        synset = "unspecified"
        extRefs = term.find("externalReferences")
        if extRefs != None:
            for extRef in extRefs.findall("externalRef"):
                resource = extRef.get("resource")
                if resource == "WordNet-eng30" or resource == "WordNet-3.0":
                    reftype = extRef.get("reftype")
                    if reftype == "synset" or reftype == "ilidef":
                        #synset = extRef.get("reference")[6:]
                        synset = extRef.get("reference")[-10:]
                        if mode == "full_dictionary":
                            if lemma not in dictionary:
                                dictionary.add(lemma)
                        elif mode == "attested_senses":
                            if lemma not in dictionary:
                                dictionary[lemma] = [synset]
                            else:
                                if synset not in dictionary[lemma]:
                                    dictionary[lemma].append(synset)
        corpus[id].extend([lemma, pos, [synset]])
    corpus = collections.OrderedDict(sorted(corpus.items()))
    if mode == "attested senses":
        for lemma in dictionary:
            dictionary[lemma] = sorted(dictionary[lemma])

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

    return sentences, dictionary

# read the contents of a folder with Semcor files in NAF-style format
def read_folder_semcor (path, lemma2synsets={}, lemma2id={}, synset2id={}, lexicon_mode="full_dictionary", mode="train", f_lex=None):

    data = []
    lemmas = set()
    for f in os.listdir(path):
        new_data = []
        if lexicon_mode == "full_dictionary":
            new_data, new_lemmas = read_file_semcor(os.path.join(path, f), "full_dictionary")
            lemmas.update(new_lemmas)
        elif lexicon_mode == "attested_senses":
            new_data, new_synsets = read_file_semcor(os.path.join(path, f), "attested_senses")
        data.extend(new_data)
        if mode == "train" and lexicon_mode == "attested_senses":
            for key, values in new_synsets.iteritems():
                if key not in lemma2synsets:
                    lemma2synsets[key] = values
                else:
                    for value in values:
                        if value not in lemma2synsets[key]:
                            lemma2synsets[key] = lemma2synsets.get(key, ()) + [value]
        #lemma2synsets.update(new_synsets)
    if mode == "train" and lexicon_mode == "full_dictionary":
    # get lexicon from the WordNet files
        lexicon = open(f_lex, "r")
        lines = lexicon.readlines()
        for line in lines:
            fields = line.split(" ")
            lemma, synsets = fields[0].strip(), fields[1:]
            # if lemma not in lemmas:
            #     continue
            for entry in synsets:
                synset = entry[:10].strip()
                if lemma not in lemma2synsets:
                    lemma2synsets[lemma] = [synset]
                else:
                    lemma2synsets[lemma].append(synset)


    if mode == "train":
        # # in case we want to remove all lemmas that correspond to just one synset
        # lemma2synsets_shrunk = {}
        # for lemma, synsets in lemma2synsets.iteritems():
        #     if len(synsets) > 1:
        #         lemma2synsets_shrunk[lemma] = synsets
        # lemma2synsets = lemma2synsets_shrunk
        lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
        index_l = 0
        index_s = 0
        for lemma, synsets in lemma2synsets.iteritems():
            lemma2id[lemma] = index_l
            index_l += 1
            for synset in synsets:
                if synset not in synset2id:
                    synset2id[synset] = index_s
                    index_s += 1
    integers = set()
    count_inst = 0
    for sentence in data:
        for word in sentence:
            lemma = word[1]
            synset = word[3]
            if synset != "unspecified":
                count_inst += 1
                if lemma in lemma2synsets:
                    if synset not in lemma2synsets[lemma]:
                        print "Synset is :" + str(synset) + " and lemma2synsets is: " + str(lemma2synsets[lemma])
                    if synset not in synset2id:
                        lemma2synsets[lemma].append(synset)
                        id = len(synset2id)
                        if id in integers:
                            print "Duplicate ID for synset " + synset + " with id " + str(id)
                        synset2id[synset] = id
                        integers.add(id)
                else:
                    lemma2synsets[lemma] = [synset]
                    if synset not in synset2id:
                        id = len(synset2id)
                        if id in integers:
                            print "Duplicate ID for synset " + synset + " with id " + str(id)
                        synset2id[synset] = id
                        integers.add(id)
                word.extend([synset2id[synset]])
            else:
                word.extend([-1])
    id2synset = {}
    id2pos = {}
    for synset, id in synset2id.iteritems():
        id2synset[id] = synset
        pos = synset.split("-")[1]
        id2pos[id] = pos
    return data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos

def get_sensekey2synset ():

    syns = list(wordnet.all_synsets())
    sensekey2synset = {}
    for syn in syns:
        synset_id = str(syn.offset())
        synset_id = (8 - len(synset_id)) * "0" + synset_id + "-" + syn.pos()
        lemmas = syn.lemmas()
        for lemma in lemmas:
            key = lemma.key()
            sensekey2synset[key] = synset_id
    with open("/home/alexander/dev/projects/BAN/neural-wsd/data/UnivRomaData/sensekey2synset.pkl", 'wb') as output:
        pickle.dump(sensekey2synset, output, pickle.HIGHEST_PROTOCOL)
    return sensekey2synset

def read_data_uniroma (path, sensekey2synset, lemma2synsets={}, lemma2id={}, synset2id={}, known_lemmas=set(),
                       synset2freq = {}, wsd_method="full_dictionary", mode="train", f_lex=None):

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
                freq = int(entry.split(":")[1])
                if freq > most_freq:
                    synset2freq[lemma] = synset
                    most_freq = freq
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
    tree = ET.parse(os.path.join(path, path_data))
    doc = tree.getroot()
    texts = doc.findall("text")
    count_inst = 0
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
                    count_inst += 1
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
        if wsd_method == "fullsoftmax":
            synset2id['notseen-n'], synset2id['notseen-v'], synset2id['notseen-a'], synset2id['notseen-r'] = 0, 1, 2, 3
            index_s = 4
        for lemma, synsets in lemma2synsets.iteritems():
            if wsd_method == "fullsoftmax" and lemma not in known_lemmas:
                continue
            lemma2id[lemma] = index_l
            index_l += 1
            for synset in synsets:
                if synset not in synset2id:
                    synset2id[synset] = index_s
                    index_s += 1
        id2synset = {}
        id2pos = {}
        for synset, id in synset2id.iteritems():
            id2synset[id] = synset
            pos = synset.split("-")[1]
            id2pos[id] = pos
    words_to_disambiguate = []
    count_ambig = 0
    count_missing1 = 0
    count_missing2 = 0
    for sentence in data:
        for word in sentence:
            if word[-1][0] != "unspecified":
                if len(word[-1]) > 1:
                    count_ambig += 1
                synsets = []
                # check if lemma is known
                if wsd_method == "fullsoftmax" and word[1] not in known_lemmas:
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
    return data, lemma2synsets, lemma2id, synset2id, id2synset, id2pos, known_lemmas, synset2freq


def format_data (wsd_method, input_data, src2id, src2id_lemmas, synset2id, seq_width, word_embedding_case,
                 word_embedding_input, sense_embeddings=None, dropword=0.0):

    inputs = []
    inputs_lemmas = []
    seq_lengths = []
    labels = []
    words_to_disambiguate = []
    # a list of the words in the sentences to be disambiguated (indexed by integers)
    indices = []
    ind_count = 0
    lemmas_to_disambiguate = []
    synsets_gold = []
    for i, sentence in enumerate(input_data):
        if len(sentence) > seq_width:
            sentence = sentence[:seq_width]
        current_input = []
        current_input_lemmas = []
        current_labels = []
        current_wtd = []
        current_gold_synsets = []
        for j, word in enumerate(sentence):
            rand_num = random.uniform(0, 1)
            if rand_num < dropword:
                continue
            current_flag = False
            if word[4][0] > -1:
                current_flag = True
            # Change depending on whether lemma or wordform is used
            if word_embedding_input == "wordform":
                if word_embedding_case == "lowercase":
                    if word[0].lower() in src2id:
                        current_input.append(src2id[word[0].lower()])
                    else:
                        current_input.append(src2id["UNK"])
                elif word_embedding_case == "mixedcase":
                    if word[0] in src2id:
                        current_input.append(src2id[word[0]])
                    else:
                        current_input.append(src2id["UNK"])
                # Changed 'word[0]' to 'word[1]' --> check difference in results
                if len(src2id_lemmas) > 0:
                    if word[1].lower() in src2id_lemmas:
                        current_input_lemmas.append(src2id_lemmas[word[1].lower()])
                    else:
                        current_input_lemmas.append(src2id_lemmas["UNK"])
            elif word_embedding_input == "lemma":
                if word[1] in src2id:
                    current_input.append(src2id[word[1]])
                else:
                    current_input.append(src2id["UNK"])
                if len(src2id_lemmas) > 0:
                    if word[1].lower() in src2id_lemmas:
                        current_input_lemmas.append(src2id_lemmas[word[1].lower()])
                    else:
                        current_input_lemmas.append(src2id_lemmas["UNK"])
            if (word[-1][0] > -1):
                current_label = np.zeros([300], dtype=float)
                if wsd_method == "similarity":
                    # TODO fix the handling of lists of synsets, like in fullmax case
                    if sense_embeddings != None:
                        for syn in word[-1]:
                            if syn < len(sense_embeddings):
                                current_label += sense_embeddings[syn]
                        current_label = current_label / len(word[-1])
                    else:
                        current_label = np.zeros(len(synset2id), dtype=int)
                        current_label[word[-1]] = 1
                elif wsd_method == "fullsoftmax":
                    current_label = np.zeros(len(synset2id), dtype=float)
                    for syn in word[-1]:
                        current_label[syn] = 1.0/len(word[-1])
                current_gold_synsets.append(word[-2])
                current_labels.append(current_label)
                indices.append(copy(ind_count))
                lemmas_to_disambiguate.append(word[1])
            # else:
            #     current_label = np.zeros(1, dtype=int)
            current_wtd.append(current_flag)
            ind_count += 1

        current_wtd += (seq_width - len(current_wtd)) * [False]
        seq_lengths.append(len(current_input))
        if (len(current_input) < seq_width):
            ind_count += seq_width - len(current_input)
            # changed [0] to [-1], should have no effect, but do check
            current_input += (seq_width - len(current_input)) * [-1]
            if len(src2id_lemmas) > 0:
                current_input_lemmas += (seq_width - len(current_input_lemmas)) * [-1]
        current_input = np.asarray(current_input)
        if len(src2id_lemmas) > 0:
            current_input_lemmas = np.asarray(current_input_lemmas)
        inputs.append(current_input)
        if len(src2id_lemmas) > 0:
            inputs_lemmas.append(current_input_lemmas)
        # extend results in a 2-d tensor where sentences are concatenated; append results in a 3-d tensor
        labels.extend(current_labels)
        synsets_gold.extend(current_gold_synsets)
        words_to_disambiguate.append(current_wtd)
    seq_lengths = np.asarray(seq_lengths)
    words_to_disambiguate = np.asarray(words_to_disambiguate)
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    inputs = np.asarray(inputs)
    inputs_lemmas = np.asarray(inputs_lemmas)

    return inputs, inputs_lemmas, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, synsets_gold


def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist


def loadGloveModel(gloveFile):

    print "Loading Glove Model"
    f = open(gloveFile,'r')
    #model = {}
    model = np.empty([400000, 300], dtype=float)
    src2id = {}
    id2src = {}
    index = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[index] = embedding
        src2id[word] = index
        id2src[id] = word
        index += 1
    print "Done.",len(model)," words loaded!"
    return model, src2id, id2src


if __name__ == "__main__":

    read_data_uniroma(path="/home/alexander/dev/projects/BAN/neural-wsd/data/UnivRomaData/WSD_Unified_Evaluation_Datasets/senseval2",
                      f_lex="/home/alexander/dev/tools/ukb_wsd/lexical_resources/lkb_sources/30/wnet30_dict.txt", lexicon_mode = "full_dictionary", mode = "train")