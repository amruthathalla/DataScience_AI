import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv(r"D:\FSDS\3march2026\Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# regressor is model and linearregression is algorithm

y_pred=regressor.predict(x_test)

comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision)

plt.scatter(x_test, y_test, color='Red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('salary of employee based on exp')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()

# how to predict future

m_coef = regressor.coef_
print(m_coef)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_coef * 20 + c_intercept
print(y_12)

# how to know model is acurate or not  -- bias is training, variance is testing

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)



y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - SSR/SST
print(r_square)

import pickle
file = 'linear_regression_model.pkl'
with open(file, 'wb') as file:
    pickle.dump(regressor, file)
print("model has been pickled and saved as linear_regression_model.pkl")