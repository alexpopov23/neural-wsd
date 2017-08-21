import _elementtree as ET
import collections
import os
import numpy as np


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
def read_folder_semcor (path):

    data = []
    lemma2synsets = {}
    for f in os.listdir(path):
        new_data, new_synsets = read_file_semcor(os.path.join(path, f))
        data.extend(new_data)
        for key, values in new_synsets.iteritems():
            if key not in lemma2synsets:
                lemma2synsets[key] = values
            else:
                for value in values:
                    if value not in lemma2synsets[key]:
                        lemma2synsets[key] = lemma2synsets.get(key, ()) + [value]
        #lemma2synsets.update(new_synsets)
    lemma2synsets_shrunk = {}
    for lemma, synsets in lemma2synsets.iteritems():
        if len(synsets) > 1:
            lemma2synsets_shrunk[lemma] = synsets
    lemma2synsets_shrunk = collections.OrderedDict(sorted(lemma2synsets_shrunk.items()))
    index = 0
    lemma2id = {}
    for lemma in lemma2synsets_shrunk.iterkeys():
        lemma2id[lemma] = index
        index += 1
    for sentence in data:
        for word in sentence:
            lemma = word[1]
            synset = word[2]
            if synset != "unspecified" and lemma in lemma2synsets_shrunk:
                if synset not in lemma2synsets_shrunk[lemma]:
                    print "Synset is :" + str(synset) + " and lemma2synsets is: " + str(lemma2synsets_shrunk[lemma])
                word.extend([lemma2synsets_shrunk[lemma].index(synset)])
            else:
                word.extend([-1])
    return data, lemma2synsets_shrunk, lemma2id

def format_data (input_data, src2id, lemma2synsets, lemma2id, seq_width):

    inputs = []
    seq_lengths = []
    labels = []
    words_to_disambiguate = []
    for i, sentence in enumerate(input_data):
        if len(sentence) > seq_width:
            sentence = sentence[:seq_width]
        current_input = []
        current_labels = []
        current_wtd = []
        for j, word in enumerate(sentence):
            current_flag = np.zeros(shape=[len(lemma2synsets)], dtype=bool)
            if word[0].lower() in src2id:
                current_input.append(src2id[word[0].lower()])
            else:
                current_input.append(src2id["UNK"])
            if word[1] in lemma2synsets:
                current_label = np.zeros(len(lemma2synsets[word[1]]), dtype=int)
                try:
                    if (word[3] < len(current_label)):
                        current_label[word[3]] = 1
                    else:
                        print "word is " + str(word) + " and current_label is " + str(current_label)
                except ValueError:
                    print "Error for word: " + str(word)
                    continue
                if word[3] > -1:
                    current_flag[lemma2id[word[1]]] = True
            else:
                current_label = np.zeros(1, dtype=int)
            current_labels.append(current_label)
            current_wtd.append(current_flag)
        current_labels += (seq_width - len(current_labels)) * [[0]]
        current_wtd += (seq_width - len(current_wtd)) * [np.zeros(shape=[len(lemma2synsets)], dtype=bool)]

        if (len(current_input) < seq_width):
            current_input += (seq_width - len(current_input)) * [0]
        current_input = np.asarray(current_input)
        inputs.append(current_input)
        seq_lengths.append(len(sentence))
        labels.append(current_labels)
        words_to_disambiguate.append(current_wtd)
    seq_lengths = np.asarray(seq_lengths)
    words_to_disambiguate = np.asarray(words_to_disambiguate)
    inputs = np.asarray(inputs)

    return inputs, seq_lengths, labels, words_to_disambiguate




