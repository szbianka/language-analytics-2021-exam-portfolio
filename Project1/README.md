# Project 1 - (Class assignment 1: Basic scripting with Python)

## Assignment description
Project 1 is equivalent to class Assignment 1 in the Language Analytics course 2021.
The purpose of the assignment was to get familiar with basic scripting in Python. More specifically, to extract specific values from various text files and assign these to its own dataframe.

## Methods
To solve the present assignment, I produced a Python script using the corpus called ‘100-english-novels’, which can be found in the ‘data’ folder in the linked GitHub repository.
First, I created a for loop to iterate over every text file found in the folder, read in their content, extract the file names and count the number of total words and unique words each novel has. The loop also collects the file names, total number of words and unique words into their own lists. These measures are further printed to the terminal. Finally, the extracted lists were collected together into a Pandas dataframe, which is saved into the output folder.

## Usage (reproducing results)
The linked GitHub repository contains:
- word_counts.py script that can be run from the command line
- a requirements.txt file listing the required Python libraries for being able to run the script
- a venv_venv.sh script for setting up a virtual environment for running the script (recommended) - NB: for running on Worker02 and MAC users
- a data folder, which contains the data used for this project
- output folder containing the output files
In order to run this script, open your terminal and:
1. Clone this repo with `git clone https://github.com/szbianka/language-analytics-2021-exam-portfolio` 
2. Navigate to the appropriate directory (Project1) 
`cd language-analytics-2021-exam-portfolio/Project1`
3. activate a virtual environment (recommended) by:
(NB: The only reason I did not include the already made virtual environment in this repository is because the file was too big on my Windows computer.)
On Windows:
If you have not used virtual environments before, you might need to run the following command first `py -m pip install --user virtualenv`
Make a folder for the virtual environment `mkdir envs`
Navigate to the env folder by `cd envs`
Create the virtual environment by `virtualenv venv`
Activate the environment by `venv\Scripts\activate`
On MAC or Worker02:
`python3 -m venv venv` 
`source venv/bin/activate`
    4. Navigate back to the appropriate folder (Project1) by `cd ..`
    5. Install the necessary libraries by `pip install -r requirements.txt`
   6. Navigate to the appropriate folder (src) by `cd src` 
   7. Run the script by: 
Windows: `python word_counts.py`
Mac:`python3 word_counts.py`
   8. Deactivate virtual environment by `deactivate`

## Discussion of results
The present assignment was designed for the purpose of making effective use of Python functions and for loops, as well as for understanding data and code structures. In particular, to get familiar with basic Python scripting (e.g. loading, saving and processing text files). Furthermore, to learn how to share these scripts with other users, which is the basis of reproducibility and collaborative coding.
