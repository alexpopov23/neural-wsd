from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd

from skip_thoughts import encoder_manager, configuration
from tensorflow.contrib.rnn.python.ops.rnn_cell import AttentionCellWrapper


# Set paths to the model.
VOCAB_FILE = "/home/lenovo/tools/skip_thoughts_models/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "/home/lenovo/tools/skip_thoughts_models/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "/home/lenovo/tools/skip_thoughts_models/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
# MR_DATA_DIR = "/dir/containing/mr/data"

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

# Load the movie review dataset.
data = ["This is a test.", "This is another test.", "And this is a test yet again.", "How about this one, bazinga!",
        "The ordeal will show your limits.", "The lunch will upset your stomach."]
# with open(os.path.join(MR_DATA_DIR, 'rt-polarity.neg'), 'rb') as f:
#   data.extend([line.decode('latin-1').strip() for line in f])
# with open(os.path.join(MR_DATA_DIR, 'rt-polarity.pos'), 'rb') as f:
#   data.extend([line.decode('latin-1').strip() for line in f])

# Generate Skip-Thought Vectors for each sentence in the dataset.
encodings = encoder.encode(data)

# Define a helper function to generate nearest neighbors.
def get_nn(ind, num=10):
  encoding = encodings[ind]
  scores = sd.cdist([encoding], encodings, "cosine")[0]
  sorted_ids = np.argsort(scores)
  print("Sentence:")
  print("", data[ind])
  print("\nNearest neighbors:")
  for i in range(1, num + 1):
    print(" %d. %s (%.3f)" %
          (i, data[sorted_ids[i]], scores[sorted_ids[i]]))

# Compute nearest neighbors of the first sentence in the dataset.
get_nn(0)