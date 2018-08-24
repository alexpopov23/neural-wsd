pos_map = {"!": ".", "#": ".", "$": ".", "''": ".", "(": ".", ")": ".", ",": ".", "-LRB-": ".", "-RRB-": ".",
           ".": ".", ":": ".", "?": ".", "CC": "CONJ", "CD": "NUM", "CD|RB": "X", "DT": "DET", "EX": "DET",
           "FW": "X", "IN": "ADP", "IN|RP": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJRJR": "ADJ", "JJS": "ADJ",
           "JJ|RB": "ADJ", "JJ|VBG": "ADJ", "LS": "X", "MD": "VERB", "NN": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
           "NNS": "NOUN", "NN|NNS": "NOUN", "NN|SYM": "NOUN", "NN|VBG": "NOUN", "NP": "NOUN", "PDT": "DET",
           "POS": "PRT", "PRP": "PRON", "PRP$": "PRON", "PRP|VBP": "PRON", "PRT": "PRT", "RB": "ADV", "RBR": "ADV",
           "RBS": "ADV", "RB|RP": "ADV", "RB|VBG": "ADV", "RN": "X", "RP": "PRT", "SYM": "X", "TO": "PRT",
           "UH": "X", "VB": "VERB", "VBD": "VERB", "VBD|VBN": "VERB", "VBG": "VERB", "VBG|NN": "VERB",
           "VBN": "VERB", "VBP": "VERB", "VBP|TO": "VERB", "VBZ": "VERB", "VP": "VERB", "WDT": "DET", "WH": "X",
           "WP": "PRON", "WP$": "PRON", "WRB": "ADV", "``": "."}
pos_normalize = {"MD|VB": "MD", "NNP|NP": "NNP", "NPS": "POS", "PR": "WRB", "NNP|VBN": "VBN", "PP": "PRP"}
pos_map_simple = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
