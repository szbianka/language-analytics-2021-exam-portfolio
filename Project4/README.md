# Project 4 - (Class assignment 6: Text classification using Deep Learning)

## Assignment description
Project 4 is equivalent to class Assignment 6 in the Language Analytics course 2021.
The purpose of the assignment was to classify scripts from the famous TV series Game of Thrones using Deep Learning. That is to say whether it is predictable which season a particular line comes from. The data can be found here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons.
 
## Methods
To solve the present task, I started by making a baseline using a 'classical' machine learning solution, which consisted of CountVectorization + LogisticRegression. The got_logistic_regression.py file contains the code for this. Later on, a  Deep learning model was further developed, using pre-trained word embeddings, tokenization, padding, a regularization model among other things as the layers of the Neural Network. Finally, the model was compiled and evaluated.
 
## Usage (reproducing results)
The linked GitHub repository contains:
got_logistic_regression.py  script that can be run from the command line
got_DL.py  script that can be run from the command line
a requirements.txt file listing the required Python libraries for being able to run the script
a got_venv.sh script for setting up a virtual environment for running the script (recommended) - NB: for running on Worker02 and MAC users
a data folder, which contains the data used for this project
utils folder with utility functions used in the script (written by Ross Deans Kristensen-McLachlan)
output folder containing the output files
In order to run this script, open your terminal and:
Clone this repo with `git clone https://github.com/szbianka/language-analytics-2021-exam-portfolio` 
Navigate to the appropriate directory (Project4) 
`cd language-analytics-2021-exam-portfolio/Project4`
activate a virtual environment (recommended) by:
(NB: The only reason I did not include the already made virtual environment in this repository is because the file was too big on my Windows computer.)
On Windows:
If you have not used virtual environments before, you might need to run the following command first `py -m pip install --user virtualenv`
Make a folder for the virtual environment `mkdir envs`
Navigate to the env folder by `cd envs`
Create the virtual environment by `virtualenv got`
Activate the environment by `got\Scripts\activate`
Navigate to the env folder by `cd envs` 
Activate the environment by `got\Scripts\activate`
On MAC or Worker02:
`python3 -m venv got` 
`source got/bin/activate`
    4. Navigate back to the appropriate folder (Project4) by `cd ..`
    5. Install the necessary libraries by `pip install -r requirements.txt`
   6. Navigate to the appropriate folder (src) by `cd src` 
+1 NB.: Please uncomment lines 41-43 in the script to get the ‘glove.6B.50d’ file the script uses in order to be able to run the code.
   7. Run the script by: 
Windows: 
`python got_logistic_regression.py`
`python got_DL.py`
			
Mac:
`python3 got_logistic_regression.py`
`python3 got_DL.py`
   8. Deactivate virtual environment by `deactivate`

## 5. Discussion of results

The task was to determine whether the lines spoken were a good predictor of season for the TV series Game of Thrones. To answer this question, I trained two models, a Logistic Regression model, which had a weighted accuracy of 0.27 and a deep learning Neural Network model, for which the weighted accuracy was 0.15. Based on these numbers it can be determined that dialogues is NOT a good predictor of season - at least the present models do not present us with such information. Instead, I propose that if one wanted to classify seasons, a different approach could be taken -e.g. the screen time of characters.
