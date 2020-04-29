from keras.preprocessing.text import Tokenizer
from data_utils import *
from create_model import create_model
import os
import pickle

def train_models(train_filepath, emotions, word2vec_location, savepath):
    df, corpus = create_dataset(train_filepath)
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(corpus)
    
    embedding_matrix = createEmbeddingMatrix(word2vec_location, tokenizer)
    
    for e in emotions:
        x_train_seq, y, input_ln = prep_data_for_model(df, corpus, e, tokenizer)
        emot_model = create_model(x_train_seq, y, embedding_matrix, input_ln)
        if not(os.path.isdir(savepath)):
            os.makedirs(savepath)

        with open(savepath + e + 'model.pkl', 'wb') as f:
            pickle.dump(emot_model, f)
