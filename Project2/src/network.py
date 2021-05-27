#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

#defining main() function
def main():
    #argument parser
    ap = argparse.ArgumentParser()
    
    #command line arguments to define
    #specifying a random-state value
    ap.add_argument("-i", 
                    "--input_file", 
                    required = False, 
                    default = "../data/pairs_df.csv",
                    type=str,
                    help = "str, Path to the input file")
    ap.add_argument("-m", 
                    "--minimum_edgeweight",
                    required=False, 
                    default=6, 
                    type=int,
                    help="int, the minimum edge weight of your choosing")
    #parse arguments 
    args = vars(ap.parse_args())

    
    #Define the input parameters 
    input_file = args["input_file"]
    minimum_edgeweight = args["minimum_edgeweight"]

    
    #load and read data into dataframe
    datapath = input_file
    #os.path.join("..", "data", "paired_df.csv")
    data = pd.read_csv(datapath)

    #extracting the mentions of persons using SpaCy
    text_entities = []

    for text in tqdm(data): 
        #create temporary list 
        tmp_entities = []
        #create doc object
        doc = nlp(text)
        # or every named entity
        for entity in doc.ents:
            #if that entity is a person
            if entity.label_ == "PERSON":
                #append to temp list
                tmp_entities.append(entity.text)
        #append temp list to main list
        text_entities.append(tmp_entities)

    #creating edgelist
    edgelist = []
    #iterate over every document in text_entities list
    for text in text_entities:
        #use itertools.combinations() to create edgelist for each combination - so for each pair of 'nodes'
        edges = list(combinations(text, 2))
        #for each combination - i.e. each pair of 'nodes'
        for edge in edges:
            #append this to final edgelist
            edgelist.append(tuple(sorted(edge)))

    #counting occurrences of each pair from the edgelist
    counted_edges = []
    for pair, value in Counter(edgelist).items():
        nodeA = pair[0]
        nodeB = pair[1]
        weight = value
        counted_edges.append((nodeA, nodeB, weight))

    #making the counted_edges list into a dataframe
    pairs_df = pd.DataFrame(counted_edges, columns=["nodeA", "nodeB", "weight"])

    #filtering pairs with more than X occcurrences
    filtered_pairs = pairs_df[pairs_df["weight"]> minimum_edgeweight] #see argparser here

    #creating network
    graph = nx.from_pandas_edgelist(filtered_pairs, 'nodeA', 'nodeB', ["weight"])

    #plotting the graph and saving figure to output/viz folder
    #defining outpath
    outpath = os.path.join("..", "output", "viz", "network.png")
    #plotting
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")

    nx.draw(graph, pos, with_labels=True, node_size=20, font_size=10)
    #saving figure
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Figure saved in output/viz folder")

    #calculating centrality measures
    #degree
    degree = nx.degree_centrality(graph)

    #betweenness
    betweenness = nx.betweenness_centrality(graph)

    #eigen vector
    eigenvec = nx.eigenvector_centrality(graph)

    #creating dataframe of these values
    centrality_measures_df = pd.DataFrame({
        'degree':pd.Series(degree),
        'betweenness':pd.Series(betweenness),
        'eigenvector':pd.Series(eigenvec)
        }).sort_values(['degree', 'betweenness', 'eigenvector'], ascending=False)

    #saving dataframe as csv file
    outpath_df = os.path.join("..", "output", "centrality_measures_df.csv")
    centrality_measures_df.to_csv(outpath_df)
    print(f"Done. Dataframe saved in output folder.")

#defining behaviour when called from command line
if __name__=="__main__":
    main()
