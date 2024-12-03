#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load the dataset
train_data = pd.read_csv("/Users/sravani/Documents/poker+hand/poker-hand-training-true.data", header=None)
test_data = pd.read_csv("/Users/sravani/Documents/poker+hand/poker-hand-testing.data", header=None)

# Rename columns
columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
train_data.columns = columns
test_data.columns = columns

# Feature Engineering: Adding new features based on rank and suit counts, flags for flush and straight
def add_features(df):
    df['is_flush'] = df.apply(lambda row: len(set([row['S1'], row['S2'], row['S3'], row['S4'], row['S5']])) == 1, axis=1)
    df['is_straight'] = df.apply(lambda row: sorted([row['C1'], row['C2'], row['C3'], row['C4'], row['C5']]) == list(range(min([row['C1'], row['C2'], row['C3'], row['C4'], row['C5']]), min([row['C1'], row['C2'], row['C3'], row['C4'], row['C5']]) + 5)), axis=1)
    return df

# Apply feature engineering to train and test data
train_data = add_features(train_data)
test_data = add_features(test_data)

# Separate features and target
X_train = train_data.drop(columns=['CLASS'])
y_train = train_data['CLASS']
X_test = test_data.drop(columns=['CLASS'])
y_test = test_data['CLASS']

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and tune the KNN model using GridSearchCV
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Retrieve the best model
best_knn_model = grid_search.best_estimator_
print("Best Parameters for KNN:", grid_search.best_params_)

# Predict on the test data
y_pred_test_knn = best_knn_model.predict(X_test_scaled)

# Evaluate the model
print("\nClassification Report for KNN Model:")
class_labels = {
    0: "Nothing in hand",
    1: "One pair",
    2: "Two pairs",
    3: "Three of a kind",
    4: "Straight",
    5: "Flush",
    6: "Full house",
    7: "Four of a kind",
    8: "Straight flush",
    9: "Royal flush"
}
classification_report_knn = classification_report(y_test, y_pred_test_knn, target_names=class_labels.values())
print(classification_report_knn)
print("\nAccuracy:", accuracy_score(y_test, y_pred_test_knn))

# Function to classify a new poker hand
def classify_poker_hand(model, hand):
    # Convert the hand into a NumPy array and scale it
    hand_array = np.array(hand).reshape(1, -1)
    hand_scaled = scaler.transform(hand_array)
    prediction = model.predict(hand_scaled)[0]
    return class_labels[prediction]


# In[2]:


# Example Predictions for All 10 Sample Hands
sample_hands = {
    "Nothing in hand": [1, 3, 2, 5, 3, 7, 4, 10, 4, 12, 0, 0],  # Random cards, no specific pattern
    "One Pair": [2, 4, 3, 4, 1, 9, 4, 6, 2, 10, 0, 0],          # One pair of 4s
    "Two Pairs": [1, 8, 3, 8, 2, 11, 4, 11, 1, 5, 0, 0],        # Two pairs: 8s and 11s
    "Three of a Kind": [2, 7, 3, 7, 1, 7, 4, 5, 2, 9, 0, 0],    # Three 7s
    "Straight": [1, 9, 2, 10, 3, 11, 4, 12, 2, 13, 0, 1],       # Straight from 9 to King
    "Flush": [1, 2, 1, 5, 1, 7, 1, 10, 1, 13, 1, 0],            # All hearts, so it's a Flush
    "Full House": [3, 5, 1, 6, 4, 6, 2, 9, 3, 9, 0, 0],         # Three 6s and two 9s
    "Four of a Kind": [2, 3, 3, 3, 1, 3, 4, 3, 1, 7, 0, 0],     # Four 3s
    "Straight Flush": [4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 1, 1],     # Straight flush in clubs from 5 to 9
    "Royal Flush": [1, 10, 1, 11, 1, 12, 1, 13, 1, 1, 1, 1]     # Royal Flush in hearts
}

feature_columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'is_flush', 'is_straight']
print("\nPredictions for Sample Hands:")
for hand_name, hand in sample_hands.items():
    predicted_hand = classify_poker_hand(best_knn_model, hand)
    print(f"{hand_name} -> Predicted: {predicted_hand}")


# In[ ]:


$

