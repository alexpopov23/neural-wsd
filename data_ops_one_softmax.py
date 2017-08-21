import _elementtree as ET
import collections
import os
import numpy as np

from copy import copy
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

npa = np.array

def build_sense_ids(data):
    words = set()
    word_to_senses = {}
    for elem in data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']

        if target_word not in words:
            words.add(target_word)
            word_to_senses.update({target_word: [target_sense]})

        else:
            if target_sense not in word_to_senses[target_word]:
                word_to_senses[target_word].append(target_sense)

    words = list(words)
    target_word_to_id = dict(zip(words, range(len(words))))
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, len(words), n_senses_from_word_id

# read a single NAF-style Semcor file
def read_file_semcor (path):
    lemma2synsets = {}
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
        id = int(term.find("span").find("target").get("id")[1:])
        synset = "unspecified"
        extRefs = term.find("externalReferences")
        if extRefs != None:
            for extRef in extRefs.findall("externalRef"):
                resource = extRef.get("resource")
                if resource == "WordNet-eng30":
                    reftype = extRef.get("reftype")
                    if reftype == "synset":
                        synset = extRef.get("reference")[6:]
                        #if synset == "06681551-n":
                        #    print "HERE"
                        if lemma not in lemma2synsets:
                            lemma2synsets[lemma] = [synset]
                        else:
                            if synset not in lemma2synsets[lemma]:
                                lemma2synsets[lemma].append(synset)
        corpus[id].extend([lemma, synset])
    corpus = collections.OrderedDict(sorted(corpus.items()))
    for lemma in lemma2synsets:
        lemma2synsets[lemma] = sorted(lemma2synsets[lemma])

    sentences = []
    current_sentence = []
    sent_counter = 1
    for word in corpus.iterkeys():
        if len(corpus[word]) == 2:
            lemma = corpus[word][1]
            corpus[word].extend([lemma, "unspecified"])
        if int(corpus[word][0]) == sent_counter:
            current_sentence.append(corpus[word][1:])
        else:
            if sent_counter != 0:
                sentences.append(current_sentence)
            sent_counter += 1
            current_sentence = []
            current_sentence.append(corpus[word][1:])
    sentences.append(current_sentence)

    return sentences, lemma2synsets

# read the contents of a folder with Semcor files in NAF-style format
def read_folder_semcor (path, lemma2synsets={}, lemma2id={}, synset2id={}, mode="train"):

    data = []
    for f in os.listdir(path):
        new_data, new_synsets = read_file_semcor(os.path.join(path, f))
        data.extend(new_data)
        if mode == "train":
            for key, values in new_synsets.iteritems():
                if key not in lemma2synsets:
                    lemma2synsets[key] = values
                else:
                    for value in values:
                        if value not in lemma2synsets[key]:
                            lemma2synsets[key] = lemma2synsets.get(key, ()) + [value]
        #lemma2synsets.update(new_synsets)
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
    for sentence in data:
        for word in sentence:
            lemma = word[1]
            synset = word[2]
            if synset != "unspecified":
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
    #DEBUG:
    integers = set()
    for synset, id in synset2id.iteritems():
        if id in integers:
            print "Duplicate ID for synset " + synset + " with id " + str(id)
        id2synset[id] = synset
        integers.add(id)
    return data, lemma2synsets, lemma2id, synset2id, id2synset

def format_data (input_data, src2id, lemma2synsets, synset2id, id2synset, seq_width, word_embedding_case,
                 word_embedding_input, sense_embeddings_path="None", similarities=None):

    inputs = []
    seq_lengths = []
    labels = []
    words_to_disambiguate = []
    # a list of the words in the sentences to be disambiguated (indexed by integers)
    indices = []
    ind_count = 0
    lemmas_to_disambiguate = []
    # structures to hold the labels for individual synsets and the sense embeddings
    if sense_embeddings_path != "None":
        label_mappings = {}
        sense_embeddings_model = KeyedVectors.load_word2vec_format(sense_embeddings_path, binary=False)
        sense_embeddings_full = sense_embeddings_model.syn0
        sense_embeddings = np.zeros(shape=(len(synset2id), 300), dtype=float)
        id2synset_embeddings = sense_embeddings_model.index2word
        #synset2id_embeddings = {}
        for i, synset in enumerate(id2synset_embeddings):
            if synset in synset2id:
                sense_embeddings[synset2id[synset]] = copy(sense_embeddings_full[i])
                #synset2id_embeddings[synset] = i
        #similarities = cosine_similarity(sense_embeddings)
        similarities = pairwise_kernels(sense_embeddings, metric="cosine")
    for i, sentence in enumerate(input_data):
        if len(sentence) > seq_width:
            sentence = sentence[:seq_width]
        current_input = []
        current_labels = []
        current_wtd = []
        for j, word in enumerate(sentence):
            current_flag = False
            if word[3] > -1:
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
            elif word_embedding_input == "lemma":
                if word[1] in src2id:
                    current_input.append(src2id[word[1]])
                else:
                    current_input.append(src2id["UNK"])
            # Only add labels for lemmas with more than 1 synset
            if word[1] in lemma2synsets:
                if (word[3] > -1):
                    if sense_embeddings_path != "None":
                        if word[3] in label_mappings:
                            current_label = copy(label_mappings[word[3]])
                        else:
                            synset_ids = [synset2id[i] for i in lemma2synsets[word[1]] if synset2id[i] != int(word[3])]
                            current_label = get_graded_label(word, similarities, id2synset, synset_ids)
                            label_mappings[word[3]] = copy(current_label)
                    else:
                        current_label = np.zeros(len(synset2id), dtype=int)
                        current_label[word[3]] = 1
                    current_labels.append(current_label)
                    indices.append(copy(ind_count))
                    lemmas_to_disambiguate.append(word[1])
            # else:
            #     current_label = np.zeros(1, dtype=int)
            current_wtd.append(current_flag)
            ind_count += 1

        #current_labels += (seq_width - len(current_labels)) * [[0]]
        current_wtd += (seq_width - len(current_wtd)) * [False]

        if (len(current_input) < seq_width):
            current_input += (seq_width - len(current_input)) * [0]
        current_input = np.asarray(current_input)
        inputs.append(current_input)
        seq_lengths.append(len(sentence))
        # extend results in a 2-d tensor where sentences are concated; append results in a 3-d tensor
        labels.extend(current_labels)
        words_to_disambiguate.append(current_wtd)
    seq_lengths = np.asarray(seq_lengths)
    words_to_disambiguate = np.asarray(words_to_disambiguate)
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    inputs = np.asarray(inputs)
    if sense_embeddings_path == "None":
        similarities = None

    return inputs, seq_lengths, labels, words_to_disambiguate, indices, lemmas_to_disambiguate, similarities

def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

def get_graded_label(word, similarities, id2synset, synset_ids):

    label = np.zeros(len(id2synset), dtype=float)
    indices = []
    for i in xrange(len(label)):
        similarity = similarities[word[3]][i]
        if word[3] == i:
            label[i] = 0.6
            continue
        if i in synset_ids:
            label[i] = 0
            continue
        if similarity > 0.5:
            indices.append(i)

        #label[i] = similarity
    if len(indices) > 0:
        p_mass = 0.4 / len(indices)
        for indx in indices:
           label[indx] = p_mass
    else:
        label[word[3]] = 1.0

    #label = softmax(label)
    return label


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



