#!/usr/bin/env python
# coding: utf-8

#libraries
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import classifier utility functions
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# matplotlib
import matplotlib.pyplot as plt


#download pretrained embedding
#get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
#get_ipython().system('unzip -q glove.6B.zip')

#defining functions - created by Ross Deans Kristensen-McLachlan 
def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
#defining main() function
def main():
    
    #Reading in data into a data frame
    filepath = os.path.join("..", "data", "Game_of_Thrones_Script.csv")
    got_data = pd.read_csv(filepath)
    #subsetting the dataframe to two columns: season and sentence 
    got_data = got_data[["Season", "Sentence"]]
    #getting the values for sentences and seasons into a list
    sentence = got_data['Sentence'].values
    season = got_data['Season'].values

    #splitting into train and test set
    X_train, X_test, y_train, y_test = train_test_split(sentence,

                                                        season, 
                                                        test_size=0.2, #splitting the data in 80% training and 20% test set
                                                        random_state=42)

    #vectorizing the data using sklearn
    vectorizer = CountVectorizer()

    #applying it to the training data
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then to the test data
    X_test_feats = vectorizer.transform(X_test)
    #creating a list of the feature names 
    feature_names = vectorizer.get_feature_names()

    #need to factorize the labels from string to a numeric value 
    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]

    # Word embeddings

    #initializing tokenizer
    tokenizer = Tokenizer(num_words=5000) #to get all the full scentences 
    #fitting to training data

    #defining tokenized training and test data
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)

    # Overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    # Padding
    #defining max length for a doc
    maxlen = 100

    #padding training data with maxlen
    X_train_pad = pad_sequences(X_train_toks,
                                padding='post', #"post" padded sequence
                                maxlen=maxlen)
    #padding test data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                                padding='post', 
                                maxlen=maxlen)

    #Using Regularization model
    l2 = L2(0.0001)

    #setting the embedding dimension to 50
    embedding_dim = 50
    #creating embedding_matrix
    embedding_matrix = create_embedding_matrix('../glove.6B.50d.txt',
                                                tokenizer.word_index, 
                                                embedding_dim)

    #building model
    model = Sequential()

    # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Embedding(vocab_size,                  
                        embedding_dim,               #Embedding input layer size
                        weights=[embedding_matrix],  #adding pretrained glove weights
                        input_length=maxlen,         # Maxlen of padded doc
                        trainable=True))             # Trainable embeddings
    model.add(Conv1D(128, 5, 
                    activation='relu',
                    kernel_regularizer=l2))          # L2 regularization 
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, activation='relu', kernel_regularizer=l2))
    model.add(Dense(1, activation='softmax'))        #softmax activation for multiple labels

    #compiling the model
    model.compile(loss='categorical_crossentropy',   #categorical_crossentropy for multiple labels
                    optimizer="adam",
                    metrics=['accuracy'])

    #printing model summary
    model.summary()

    # Create history of the model
    history = model.fit(X_train_pad, y_train,
                        epochs=4,
                        verbose=False,
                        validation_data=(X_test_pad, y_test),
                        batch_size=10)

    #evaluating the model 
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    #plotting the history
    plot_history(history, epochs = 4)

    #create predictions
    predictions = model.predict(X_test_pad, batch_size = 10)
    #printing the classification report
    class_rep = classification_report(y_test, predictions.argmax(axis=1))
    print(class_rep)
    #saving the output as a txt file
    outpath = os.path.join("..", "output", "got_DL_model")
    file = open(outpath, "w")
    file.write(class_rep)
    file.close()

#defining behaviour when called from command line
if __name__=="__main__":
    main()