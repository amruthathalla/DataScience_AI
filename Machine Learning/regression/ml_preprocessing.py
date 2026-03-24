#Import Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
dataset = pd.read_csv(r"D:\FSDS\2march2026\Data.csv")


# Split features (X) and target (y) - Independent & Dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle missing values - Replaces missing numbers with column average
imputer = SimpleImputer(strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Convert Text to Numbers (OneHotEncoding)

# Encode categorical independent variable
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Encode dependent variable - Encode target (Yes/No → 0/1)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split into training & testing data  -  Train on old data, test on unseen data
#	80% → Training 
#	20% → Testing 
# X → input data (State, Age, Salary)
# y → output (Purchased: Yes/No)
#random_state - It keeps the split same every time you run. If removed → split changes every run.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Feature Scaling
# Salary is much bigger. Machine Learning models get confused. So we scale values to similar range.

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
