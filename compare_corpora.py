'''
A script to compare the level of overlap between different corpora, e.g. SemCor and SenseEval
'''

import _elementtree as ET
import os

path_corpus1 = "/home/lenovo/dev/neural-wsd/data/wsd_corpora-master/semcor3.0/all"
path_corpus2 = "/home/lenovo/dev/neural-wsd/data/wsd_corpora-master/senseval2"

synsets_corpus1 = set()
synsets_corpus2 = set()
synset_corpora = [synsets_corpus1, synsets_corpus2]
sense_eval_all_cases = 0
sense_eval_common_cases = 0
for i, path in enumerate([path_corpus1, path_corpus2]):
    for f in os.listdir(path):
        filepath = os.path.join(path, f)
        tree = ET.parse(filepath)
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
                            synset_corpora[i].add(synset)
                            if i == 1:
                                sense_eval_all_cases += 1
                                if synset in synsets_corpus1:
                                    sense_eval_common_cases += 1

common_synsets = synsets_corpus1.intersection(synsets_corpus2)
print "Unique synsets in SemCor: " + str(len(synsets_corpus1))
print "Unique synsets in SenseEval: " + str(len(synsets_corpus2))
print "Common synsets for SemCor and SenseEval: " + str(len(common_synsets))

print "Number of all senseval cases: " + str(sense_eval_all_cases)
print "Number of senseval cases attested in SemCor as well: " + str(sense_eval_common_cases)

