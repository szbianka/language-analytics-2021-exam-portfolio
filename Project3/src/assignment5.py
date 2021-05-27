#!/usr/bin/env python
# coding: utf-8

#importing packages
import os
import sys
sys.path.append(os.path.join(".."))
import utils.classifier_utils_a5 as clf
from sklearn import metrics

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.model_selection import train_test_split
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#defining main() function
def main():
    
    #reading in data set to a dataframe (which I previously merged)
    filename = os.path.join("..", "data", "full_corona.csv")

    data = pd.read_csv(filename, index_col=0)

    #subsetting data to include only the tweets and sentiment columns
    data = data[["OriginalTweet", "Sentiment"]]

    #rename "Exrtemely Negative" and "Negative" Sentiment to 0
    #rename "Exrtemely Positive" and "Positive" Sentiment to 1
    data["Sentiment"].replace({"Extremely Negative": "0", "Negative": "0","Extremely Positive": "1", "Positive": "1" }, inplace=True)

    #remove "Neutral" sentiment from dataset
    #getting the indexes of Neutral values
    neutral = data[data["Sentiment"] == "Neutral"].index
    #deleting these rows
    data.drop(neutral, inplace=True)
    #converting to appropriate format
    data["Sentiment"]= data["Sentiment"].astype(float)
    data["OriginalTweet"]= data["OriginalTweet"].astype(str)

    #creating more balanced data with 5000 random samples for each sentiment
    data = clf.balance(data, 5000)

    ##Preprocessing tweets ##
    #making text lower case
    data["OriginalTweet"] = data["OriginalTweet"].str.lower()

    #removing stop words
    #make set of stopwords
    stopwords_set = set(stopwords.words('english'))

    #defining function for removing stopwords
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in stopwords_set])

    #removing stopwords
    data["OriginalTweet"] = data["OriginalTweet"].apply(lambda text: remove_stopwords(text))

    #removing punctuations
    punctuations = string.punctuation

    #defining function for removing punctuations
    def remove_punct(text):
        trans = str.maketrans('', '', punctuations)
        return text.translate(trans)

    #removing punctuations from data
    data["OriginalTweet"] = data["OriginalTweet"].apply(lambda text: remove_punct(text))

    #removing mentions with function
    def remove_mentions(text):
        return re.sub('@[^\s]+', ' ', text)

    data["OriginalTweet"] = data["OriginalTweet"].apply(lambda text: remove_mentions(text))

    #removing URLs with function
    def remove_urls(text):
        return re.sub('((www\.[^\s]+) | (https?://[^\s]+))', ' ', text)

    data["OriginalTweet"] = data["OriginalTweet"].apply(lambda text: remove_urls(text))

    #tokenizing the tweets to get separate words
    tokenizer = RegexpTokenizer(r"\w+")
    data["OriginalTweet"] = data["OriginalTweet"].apply(tokenizer.tokenize)

    #lemmatizing with function
    lemma = nltk.WordNetLemmatizer()

    def lemmatizer(data):
        txt = [lemma.lemmatize(word) for word in data]
        return data

    data["OriginalTweet"] = data["OriginalTweet"].apply(lambda text: lemmatizer(text))

    #creating lists with the variables from the dataframe
    texts = data["OriginalTweet"]
    labels = data["Sentiment"]

    X=data.OriginalTweet
    y=data.Sentiment

    max_len = 500
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=0.2, random_state=2)

    #defining model
    def NN_model():
        inputs = Input(name='inputs',shape=[max_len])#defining input
        layer = Embedding(2000,50,input_length=max_len)(inputs) #applyig embeddings
        layer = LSTM(64)(layer) #long-short term memory layer
        layer = Dense(256,name='FC1')(layer) #dense layer
        layer = Activation('relu')(layer) #relu activation function
        layer = Dropout(0.5)(layer) #dropping a few neurons
        layer = Dense(1,name='out_layer')(layer) #dense layer
        layer = Activation('sigmoid')(layer) #relu activation function
        model = Model(inputs=inputs,outputs=layer) #final output of the model
        return model #return model

    #training model
    model = NN_model() #setting model to our defined model
    model.compile(loss= "binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

    history=model.fit(X_train,Y_train,batch_size=80,epochs=4, validation_split=0.1)#training data
    print(f"Model training finished!")

    #evaluating model
    #training accuracy
    accuracy = model.evaluate(X_train,Y_train)
    print("Training Accuracy: {:0.2f}".format(accuracy[1]))
    #testing accuracy
    accuracy = model.evaluate(X_test,Y_test)
    print("Testing Accuracy: {:0.2f}".format(accuracy[1]))

    #predicting the test data
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5) #above 0,5 is positive sentiment, below is negative

    #printing the evaluation to the terminal
    classifier_metrics = metrics.classification_report(Y_test, y_pred)
    print(classifier_metrics)

    #saving the output as a txt file
    outpath = os.path.join("..", "output", "confusion_matrix.txt")
    file = open(outpath, "w")
    file.write(classifier_metrics)
    file.close()

    #plotting confusion matrix
    CR= confusion_matrix(Y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=CR, figsize =(10, 10),
                         show_absolute= True,
                         show_normed = True)
    #showing plot
    plt.show()
    #saving figure to output folder
    fig.savefig("../output/confusion_matrix.png")
    plt.close(fig) 

#defining behaviour when called from command line
if __name__=="__main__":
    main()