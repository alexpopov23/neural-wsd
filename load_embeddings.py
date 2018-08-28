import os

import gensim
import numpy

def load (embeddings_path, binary=False):
    """Loads an embedding model with gensim

    Args:
        embeddings_path: A string, the path to the model
    Returns:
        embeddings: A list of vectors
        src2id: A dictionary, maps strings to integers in the list
        id2src: A dictionary, maps integers in the list to strings
    """
    _, extension = os.path.splitext(embeddings_path)
    if extension == ".txt":
        binary = False
    elif extension == ".bin":
        binary = True
    embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=binary)
    embeddings = embeddings_model.syn0
    id2src = embeddings_model.index2word
    src2id = {}
    for i, word in enumerate(id2src):
        src2id[word] = i
    if "UNK" not in src2id:
        if "unk" in src2id:
            src2id["UNK"] = src2id["unk"]
            id2src[src2id["UNK"]] = "UNK"
            del src2id["unk"]
        else:
            unk = numpy.zeros(len(embeddings[0]))
            src2id["UNK"] = len(src2id)
            embeddings = numpy.concatenate((embeddings, [unk]))
    return embeddings, src2id, id2src