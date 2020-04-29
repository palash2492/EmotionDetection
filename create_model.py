from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Bidirectional, GRU, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
import numpy as np

def create_model(x_train_seq, y, embedding_matrix, input_ln, max_words=40000,numGRUunits=16, numFCunits=128, numepochs=5, batch_size=32, drpout=0.5):
    model_ptw2v = Sequential()
    e = Embedding(max_words, embedding_matrix.shape[1], weights=[embedding_matrix], input_length=input_ln, trainable=False)
    model_ptw2v.add(e)
    
    model_ptw2v.add(Bidirectional(GRU(numGRUunits, return_sequences=True)))
    model_ptw2v.add(GlobalAveragePooling1D())
    
    model_ptw2v.add(Dense(numFCunits, activation='relu'))
    model_ptw2v.add(Dropout(drpout))
    model_ptw2v.add(Dense(1, activation='sigmoid'))
    
    model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_ptw2v.fit(x_train_seq, y,epochs=numepochs, batch_size=batch_size, verbose=2)
    
    return model_ptw2v