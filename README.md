# Poker Hand Classifier

## Project Overview
This project aims to classify poker hands using various machine learning algorithms. Given a set of cards represented in a dataset, we implement and compare different classifiers to determine the hand's rank.

## Features
- *Data Preprocessing*: Prepare the poker hand data for training and testing the models.
- *Model Implementation*: Utilize different machine learning algorithms:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
- *Performance Evaluation*: Compare the accuracy and efficiency of each implemented model.

## Technologies Used
- *Python*: Programming language.
- *Scikit-learn*: Machine learning library for Python.
- *Pandas*: Data manipulation and analysis library.

## Getting Started
Follow these steps to set up the project on your local machine:

First, clone the repository to your local machine, then run the following commands:
   ```bash
   git clone https://github.com/shashisingarapu0607/CMPE255-FinalProject-PokerHand.git
  cd CMPE255-FinalProject-PokerHand

  pip install -r requirements.txt
  Load the datset path as pd.read_csv(path)
  python KNN.py
  python NaiveBayes.py
  python RandomForest.py

After running these models, the predicted hands will be displayed.



