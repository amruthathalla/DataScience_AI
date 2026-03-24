# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:02:14 2026

@author: Amrutha Thalla
"""
# What is logistic regression?
# It's a classification algorithm — it learns to predict yes or no (will someone buy or not?). Internally it calculates a probability between 0 and 1, then applies a threshold (usually 0.5): above → Purchased, below → Not purchased.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# pipeline
# Step 1 :Load data---400 rows, columns: Age, Salary, Purchased
# Step 2 :Split 75 / 25--300 rows to train · 100 rows to test
# Step 3 :Scale features--StandardScaler: mean=0, std=1 so Age & Salary are comparable
# Step 4 :Train model--LogisticRegression fits a decision boundary
# Step 5 :Evaluate---Confusion matrix, accuracy, precision, recall, AUC


dataset = pd.read_csv(r'C:/Users/Amrutha Thalla/FSDS/DataScience_AI/Machine Learning/classifications/logistic_reg/logit classification.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# scale features
# Without scaling, Age (20–60) and Salary (15k–150k) are very different scales. The model unfairly weighs Salary more.
# Before scaling --Age: 35, Salary: 72,000

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# After StandardScaler, both features are centered around 0 with equal spread. The model treats them fairly.
# After scaling --Age: 0.42, Salary: 0.38

#datapreprocessing done 

#******************************************************************************************

# Next step is we are going to build the logistic model  
# apply this model into our dataset 
# This is linear model library thats why we called from sklear.linear_model


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# we build our logistic model and fit it to the training set & we predict our test set result 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(X_train, y_train)
print(bias)

variance = classifier.score(X_test, y_test)
print(variance)


#-----------------FUTURE PREDICTION ------------


dataset1 = pd.read_csv(r"C:/Users/Amrutha Thalla/Downloads/Future prediction1.csv")
d2 = dataset1.copy()
dataset1 = dataset1.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv')


import os
os.getcwd()

#-------------------------------------------------------------------------

from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#---------------------------------------------------------------------
# evaluation metrics 

# Accuracy - 89%
# Out of 100 people, 89 were predicted correctly.

# Precision- 89%
# For class 1 (buyers): 0.89 → out of everyone it predicted as "buyer", 89% actually bought.

# recall
# For class 1 (buyers): 0.75 → the model found only 75% of real buyers, missing 25%.

# f1 score
# A single number combining both. Useful when classes are imbalanced (68 vs 32 here).
