#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1- Basic scripting with Python

#importing libraries
import os
from pathlib import Path
import pandas as pd

#defining main() function
def main():
    
    #defining filepath
    filepath = os.path.join("..", "data")

    #making for loop for extracting the titles of the novels
    #creating empty lists for the title, number of words and number unique words
    title_list=[]
    word_list = []
    unique_word_list = []

    #for loop for iterating over every document that ends with .txt
    for filename in Path(filepath).glob("*.txt"):
        with open(filename, "r", encoding= "utf-8") as file:
            #loading the files and reading their content
            loaded_text = file.read()
            #making a list of filenames without including the path
            file_list = os.path.basename(filename)
            #appending the empty list with the titles
            title_list.append(file_list)

            #counting total number of words
            #by splitting on whitespace to define separate words
            words = loaded_text.split()
            #counting total number of words
            total_words = len(words)
            #counting unique words
            unique_words = set(words)
            #appending the empty list with the length of words
            word_list.append(len(words))
            #appending the empty list with the length of unique words
            unique_word_list.append(len(unique_words))
            #printing the number of words and unique words each novel has to the terminal
            print(f" {file_list} contains {total_words} words and {len(unique_words)} unique words.")

    #making dataframe with the extracted lists
    df = pd.DataFrame({
        'filename':title_list,
        'total_words':word_list,
        'unique_words':unique_word_list})


    # __Writing file__

    #saving dataframe as a .csv file in output folder
    outpath = os.path.join("..", "output", "word_counts_df.txt")
    df.to_csv(outpath, index = False)
    print(f"Done. Dataframe written to output folder.")

#defining behaviour when called from command line
if __name__=="__main__":
    main()
