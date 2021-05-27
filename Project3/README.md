# Project 3 - (Class assignment 5: Supervised Machine Learning)

## Assignment description
Project 3 is equivalent to class Assignment 5 in the Language Analytics course 2021.
For this assignment I chose the Coronavirus Tweets NLP- text classification dataset from Kaggle (can be seen here: https://www.kaggle.com/datatattle/covid-19-nlp-text-classification). The purpose of the assignment was to see if it was possible to predict sentiment Positive or Negative) with supervised machine learning from coronavirus tagged tweets from across the world.
## Methods
Before applying supervised machine learning to the dataset, several preprocessing steps were implemented. First, even though the original dataset was already split into train and test sets, which came in two csv files, I merged these back together in order to be able to split the data differently if I wanted to. The merged dataset is included in the data folder in the GitHub repository. Furthermore, I set “Extremely Negative” and “Negative” sentiments to 0, and “Extremely Positive'' and “Positive” sentiments to 1, while filtering “Neutral” sentiment away. This was done with the purpose of later on making the building of the model easier. Then, preprocessing of the tweets followed. The steps included making the text lower case, removing stop words, punctuations, mentions and URLs. Then, tokenization and lemmatization was done on them. Afterwards, I drew a random sample of 5000 tweets for each sentiment, so that the data would be more balanced. Next, the data was split into training and test datasets on a 80-20% ratio. Thereafter, I trained a Neural Network model, which was then fitted to the test data and then I finally predicted and evaluated the model.

## Usage (reproducing results)
The linked GitHub repository contains:
assignment5.py script that can be run from the command line
a requirements.txt file listing the required Python libraries for being able to run the script
a venv_venv.sh script for setting up a virtual environment for running the script (recommended) - NB: for running on Worker02 and MAC users
a data folder, which contains the data used for this project
utils folder with utility functions used in the script (written by Ross Deans Kristensen-McLachlan)
output folder containing the output files
In order to run this script, open your terminal and:
Clone this repo with `git clone https://github.com/szbianka/language-analytics-2021-exam-portfolio` 
Navigate to the appropriate directory (Project3) 
`cd language-analytics-2021-exam-portfolio/Project3`
activate a virtual environment (recommended) by:
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
    4. Navigate back to the appropriate folder (Project3) by `cd ..`
    5. Install the necessary libraries by `pip install -r requirements.txt`
   6. Navigate to the appropriate folder (src) by `cd src` 
   !/have to download nltk.download('wordnet') before running the script
   8. Run the script by: 
Windows: `python assignment5.py`
Mac:`python3 assignment5.py`
   8. Deactivate virtual environment by `deactivate`

## 5. Discussion of results
As mentioned earlier, I wanted to study if it was possible to predict sentiment from coronavirus tagged tweets from Twitter. After preprocessing the data and playing around with some parameters, the final model I trained (with a batch size of 80 and 4 epochs) had a weighted accuracy of 0.77, which implies that it is possible to predict sentiment with a decent accuracy - however it could be more accurate.
