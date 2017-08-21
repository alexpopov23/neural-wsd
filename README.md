# neural-wsd
Neural network with LSTMs that performs WSD with respect to WordNet senses.


# files to use:
- data_ops_final for data transformations
- neural_wsd_embedding_similairity for wsd with vector comparison at final step
- neural_wsd_single_softmax for wsd using a large softmax with all possible synsets at final step

# sample command:
-wsd_method similarity -word_embedding_method glove -word_embedding_input wordform -word_embeddings_src_path /home/lenovo/dev/word-embeddings/glove.6B/glove.6B.300d.txt -word_embedding_dim 300 -word_embedding_case lowercase -sense_embeddings_src_path /home/lenovo/dev/word-embeddings/sense-embeddings/WN30WNGWN30glConOneGraphRelSCOne_embeddings.bin -learning_rate 0.1 -training_iterations 100001 -batch_size 100 -n_hidden 100 -sequence_width 50 -training_data /home/lenovo/dev/neural-wsd/data/wsd_corpora-master/semcor3.0/all -lexicon /home/lenovo/tools/ukb_wsd/lkb_sources/wn30.lex