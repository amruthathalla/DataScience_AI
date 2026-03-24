# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:34:17 2026

@author: Amrutha Thalla
"""

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


# importing the dataset

dataset = pd.read_csv(r'C:/Users/Amrutha Thalla/FSDS/DataScience_AI/Machine Learning/classifications/logistic_reg/logit classification.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# scale features
# Without scaling, Age (20–60) and Salary (15k–150k) are very different scales. The model unfairly weighs Salary more.
# Before scaling --Age: 35, Salary: 72,000

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training the svm model on the training set
from sklearn.svm import SVC 
classifier = SVC()
classifier.fit(X_train,y_train)

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





