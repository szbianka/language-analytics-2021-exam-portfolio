# Project 2 - (Class assignment 4: Network analysis)

## Assignment description
Project 2 is equivalent to class Assignment 4 in the Language Analytics course 2021.
The purpose of the assignment was to create a reusable command-line network analysis pipeline.
The script takes any given dataset (weighted edge list) as an input -provided that the edge list is saved as a .CSV with the column headers "nodeA", "nodeB" and includes how many times they appear together - and builds networks based on entities appearing together in the same documents.

## Methods
To solve the given task, I created a Python script that can be run from the command-line based on the work we did in class. First, the given dataset was filtered by setting a threshold for how many times each pair (node) occurs together in the dataset. Then, using the data, the script creates a connectivity network and also visualizes it. The visualization was saved into the output/viz folder. Finally, the centrality measures (degree, betweenness and eigenvector) for each node were calculated and saved into a dataframe. The data frame was also saved into the output folder.

## Usage (reproducing results)
The linked GitHub repository contains:
- network.py script that can be run from the command line
- a requirements.txt file listing the required Python libraries for being able to run the script
- a venv_venv.sh script for setting up a virtual environment for running the script (recommended) - NB: for running on Worker02 and MAC users
- a data folder, which contains the data used for this project
- utils folder with utility functions used in the script (written by Ross Deans Kristensen-McLachlan)
- output folder containing the output files
In order to run this script, open your terminal and:
1. Clone this repo with `git clone https://github.com/szbianka/language-analytics-2021-exam-portfolio` 
2. Navigate to the appropriate directory (Project2) 
`cd language-analytics-2021-exam-portfolio/Project2`
3. activate a virtual environment (recommended) by:
(NB: The only reason I did not include the already made virtual environment in this repository is because the file was too big on my Windows computer.)
On Windows:
If you have not used virtual environments before, you might need to run the following command first `py -m pip install --user virtualenv`
Make a folder for the virtual environment `mkdir envs`
Navigate to the env folder by `cd envs`
Create the virtual environment by `virtualenv venv`
Activate the environment by `venv\Scripts\activate`
Navigate to the env folder by `cd envs` 
Activate the environment by `venv\Scripts\activate`
On MAC or Worker02:
`python3 -m venv venv` 
`source venv/bin/activate`
    4. Navigate back to the appropriate folder (Project2) by `cd ..`
    5. Install the necessary libraries by `pip install -r requirements.txt`
   6. Navigate to the appropriate folder (src) by `cd src` 
NB:  !python -m spacy download en_core_web_sm needs to be run before running my script!
   7. Run the script by: 
Windows: `python network.py`
Mac:`python3 network.py`
   8. Deactivate virtual environment by `deactivate`

## 5. Discussion of results
The present project was created with the purpose of creating reusable and reproducible pipelines for data analysis that can be executed from the command-line. Moreover, to show that it is possible to perform network analysis, using networkx, from a given text, as the .CSV the script takes as input can be created from almost any data that includes text.
