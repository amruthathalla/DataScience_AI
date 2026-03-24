# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:29:39 2026

@author: Amrutha Thalla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Amrutha Thalla\Downloads\emp_sal.csv")

# Separating Features and Target
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear Regression Model -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Prediction
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)
# [330378.78787879]

# Linear Regression Visualization
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Linear regression Graph')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Polynomial Regression
# Linear regression cannot capture curves, So we convert features to polynomial features.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

#Train Polynomial Model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X,y,color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('truth or bluff (poly reg)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Polynomial Prediction
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)
# [174878.07765106]


# Support Vector Regression (SVR)
from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree = 5,gamma = 'scale' ) 
svr_regressor.fit(X,y)

# SVR requires feature scaling, Otherwise predictions may be inaccurate.
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1,1))

svr_regressor.fit(X_scaled, y_scaled)

# SVR Prediction
svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred) 
# [164079.01344549]

# KNN Regression Model
from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=2)

# Train KNN Model
knn_reg_model.fit(X,y)

#  KNN Prediction
knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred) 
# [175348.8372093]

# Decision tree regression
from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)


#Random forest algorithm
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=20,random_state=0)
rf_reg.fit(X,y)

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)

# score comparision
# Linear Regression Prediction
print(lin_model_pred) # [330378.78787879]

print('*************************')
# Polynomial Prediction
print(poly_model_pred) # [174878.07765106]

print('*************************')
# SVR Prediction
print(svr_model_pred) # [164079.01344549]

print('*************************')
#KNN Regression Prediction
print(knn_reg_pred) # [175348.8372093]

print('*************************')
#Desicion tree algorithm
print(dt_pred) # [150000.]

