import numpy as np
import pandas as pd
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

max_words = 40000
def cleanTweet(x):
    temp=x
    modified=re.search('@[0-9a-zA-Z_]+',temp)
    while modified!=None:
        temp=re.sub(modified.group(0),'',temp)
        modified=re.search('@[0-9a-zA-Z_]+',temp)
    
    Tweet=re.sub('[+!#?@,.:";\']', '', temp).lower()
    Tweet=re.sub(r'\\n', ' ', Tweet)
    Tweet=''.join([x for x in Tweet if x in string.printable])
    return Tweet.strip()
	
def create_dataset(filepath):
    df = pd.read_csv(filepath,sep='\t',skiprows=None,header=(0))
    df['cleanTweet'] = df['Tweet'].apply(lambda x: cleanTweet(x))
    
    corpus=[x for x in df['cleanTweet']]
    
    return df,corpus

def prep_data_for_model(df,corpus,emotion, tokenizer):
    sequences = tokenizer.texts_to_sequences(corpus)
    
    length = []
    for x in corpus:
        length.append(len(x.split()))
    
    input_ln=max(length)*2
    x_train_seq = pad_sequences(sequences, maxlen=input_ln)
    y=df[emotion]
    return x_train_seq, y, input_ln
    
def createEmbeddingMatrix(word2vec_location, tokenizer):
    model = KeyedVectors.load_word2vec_format(word2vec_location, binary=True)
    
    embeddings_index = {}
    for w in model.vocab.keys():
        embeddings_index[w] = model[w]
    index2word=model.wv.index2word
    
    embedding_matrix = np.zeros((max_words, 300))
    for word, i in tokenizer.word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix