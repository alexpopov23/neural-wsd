import xml.etree.ElementTree as ET
import tensorflow as tf

import collections
import os
import copy

def read_data_semcor_orig (path):
    tree = ET.parse(path)
    doc = tree.getroot()
    input_format = []
    context = doc.find("context")
    paragraphs = context.findall("p")
    for para in paragraphs:
        sentences = para.findall("s")
        for sent in sentences:
            sent_format = []
            wfs = sent.iter()
            for wf in wfs:
                if wf is sent:
                    continue
                word = wf.text
                lemma = wf.get("lemma")
                wordnet_id = wf.get("lexsn")
                triple = (word, lemma, wordnet_id)
                sent_format.append(triple)
            input_format.append(sent_format)

    return input_format

def read_file_semcor (path):
    seen_lemmas = set()
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
        seen_lemmas.add(lemma)
        corpus[id].extend([lemma, synset])
    corpus = collections.OrderedDict(sorted(corpus.items()))

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

    return sentences, seen_lemmas

def read_folder_semcor (path):

    data = []
    seen_lemmas = set()
    for f in os.listdir(path):
        new_data, new_synsets = read_file_semcor(os.path.join(path, f))
        data.extend(new_data)
        seen_lemmas.update(new_synsets)
    return data, seen_lemmas

def init_weight_matrices_for_lexicon (path_to_lexicon, n_hidden, src2id, seen_lemmas):

    lexicon = open(path_to_lexicon, "r")
    lemma2synset = {}
    weights_biases = {}
    for line in lexicon.readlines():
        entries = line.split()
        lemma = entries[0]
        if lemma not in seen_lemmas:
            continue
        synsets = [entry[:10] for entry in entries[1:]]
        if lemma in src2id:
            lemma_id = src2id[lemma]
        else:
            continue
        lemma2synset[lemma] = synsets
        #TODO figure out a good initialization
        w = tf.Variable(tf.random_normal([2*n_hidden, len(synsets)]), name="w-"+str(lemma_id))
        b = tf.Variable(tf.random_normal([len(synsets)]), name="b-"+str(lemma_id))
        weights_biases["w-"+str(lemma_id)] = w
        weights_biases["b-"+str(lemma_id)] = b
    weights_biases["w-unspecified"] = tf.zeros([2*n_hidden,1])
    weights_biases["b-unspecified"] = tf.zeros([0])

    return lemma2synset, weights_biases

if __name__ == "__main__":
    #read_folder_semcor("/home/lenovo/dev/neural-wsd/data/wsd_corpora-master/semcor3.0/brownv/")
    #read_file_semcor("/home/lenovo/dev/neural-wsd/data/wsd_corpora-master/semcor3.0/brownv/br-a03.naf")
    init_weight_matrices_for_lexicon("/home/lenovo/tools/ukb_wsd/lkb_sources/wn30.lex")

