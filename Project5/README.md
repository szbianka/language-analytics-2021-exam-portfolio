# Project 5 (self-assigned)- Video games

## Assignment description
Project 5 is a self-assigned project in the Language Analytics course 2021.
The purpose of the project was to - similarly to the previous project - see whether genre is a good predictor of gaming platform. That is to say, whether the tools one plays with also determines a preference for the genre of the games.

The data used in the project was downloaded from here: https://www.kaggle.com/gregorut/videogamesales 

## Methods
To investigate whether genre is a good predictor of gaming platform, I built a 'classical' machine learning model, which consisted of CountVectorization +  LogisticRegression. The reason for this, is among other things, that from my experience during the course, the Logistic Regression models perform almost as well as Convolutional Neural Networks, not to mention that they are computationally more cost effective as well. Therefore I wanted to reflect a better ‘real life’ solution for the classification of this data.

## Usage (reproducing results)
The linked GitHub repository contains:
- vid_game_logistic_regression.py script that can be run from the command line
- a requirements.txt file listing the required Python libraries for being able to run the script
- a venv_venv.sh script for setting up a virtual environment for running the script (recommended) - NB: for running on Worker02 and MAC users
- a data folder, which contains the data used for this project
- utils folder with utility functions used in the script (written by Ross Deans Kristensen-McLachlan)
- output folder containing the output files
In order to run this script, open your terminal and:
1. Clone this repo with `git clone https://github.com/szbianka/language-analytics-2021-exam-portfolio` 
2. Navigate to the appropriate directory (Project1) 
`cd language-analytics-2021-exam-portfolio/Project5`
3. activate a virtual environment (recommended) by:
(NB: The only reason I did not include the already made virtual environment in this repository is because the file was too big on my Windows computer.)
On Windows:
If you have not used virtual environments before, you might need to run the following command first `py -m pip install --user virtualenv`
Make a folder for the virtual environment `mkdir envs`
Navigate to the env folder by `cd envs`
Create the virtual environment by `virtualenv venv`
Activate the environment by `venv\Scripts\activate`
On MAC or Worker02:
`python3 -m venv edge` 
`source edge/bin/activate`
    4. Navigate back to the appropriate folder (Project5) by `cd ..`
    5. Install the necessary libraries by `pip install -r requirements.txt`
   6. Navigate to the appropriate folder (src) by `cd src` 
   7. Run the script by: 
Windows: `python vid_game_logistic_regression.py`
Mac:`python3 vid_game_logistic_regression.py`
   8. Deactivate virtual environment by `deactivate`

## 5. Discussion of results
Just as it can be seen from the previous assignment, based on the model output numbers it can be determined that genre is NOT a good predictor of platform - at least the present models do not present us with such information. Instead, perhaps different variables should have been investigated.
