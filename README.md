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

# Small sample of the test data for evaluation
# Sample hands to test with feature names
sample_hands = {
    "Nothing in hand": [1, 3, 2, 5, 3, 7, 4, 10, 4, 12, 0, 0],  # Random cards, no specific pattern
    "One Pair": [2, 4, 3, 4, 1, 9, 4, 6, 2, 10, 0, 0],          # One pair of 4s
    "Two Pairs": [1, 8, 3, 8, 2, 11, 4, 11, 1, 5, 0, 0],        # Two pairs: 8s and 11s
    "Three of a Kind": [2, 7, 3, 7, 1, 7, 4, 5, 2, 9, 0, 0],    # Three 7s
}

feature_columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'is_flush', 'is_straight']

# Test each sample hand and print the prediction with feature names
print("\nPredictions for Various Sample Hands:")
for hand_name, hand in sample_hands.items():
    hand_df = pd.DataFrame([hand], columns=feature_columns)  # Add feature names
    predicted_hand = classify_poker_hand(best_rf_model, hand_df.values[0])
    print(f"Test Hand: {hand_name} -> Predicted: {predicted_hand}")

