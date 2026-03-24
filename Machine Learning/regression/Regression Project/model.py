# What is Supervised Learning?
# Supervised learning is a type of machine learning where you teach a model using labeled data. That means you have examples that already include the correct answer. The model learns the relationship between the input (features) and the output (target). After training, it can predict the output for new, unseen inputs.There are two main types: Regression: The output is a continuous number (e.g., house price, temperature). Classification: The output is a category (e.g., spam or not spam, type of flower). In this project, we are doing regression – predicting the price of a house based on several features.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# The Dataset: USA_Housing.csv
# The dataset contains information about houses in the USA
# Load and Prepare the Data
data = pd.read_csv(r"C:\Users\Amrutha Thalla\FSDS\DataScience_AI\Machine Learning\Regression Project\USA_Housing.csv")

# Preprocessing
X = data.drop(['Price', 'Address'], axis=1) 
y = data['Price']
# X contains all input features (the ones listed above).
# y is the price we want to predict.
# We drop Address because it's text and not useful for numerical prediction (at least in this basic model).

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# We split the data so that 80% is used to train the models and 20% is kept aside to test how well they perform on unseen data.
# random_state=0 ensures we get the same split every time (for reproducibility).

# Define models
models = {
    'LinearRegression': LinearRegression(), # The simplest: assumes a straight‑line relationship between features and price.
    'RobustRegression': HuberRegressor(), # Similar to linear regression but less sensitive to outliers (extreme values).
    'RidgeRegression': Ridge(), #  Linear regression with regularization (penalty) to prevent overfitting.
    'LassoRegression': Lasso(), # Another regularized linear model that can also reduce some feature weights to zero (feature selection).
    'ElasticNet': ElasticNet(), # Combines Ridge and Lasso penalties.
   'PolynomialRegression': Pipeline([
    ('poly', PolynomialFeatures(degree=4)),
    ('linear', LinearRegression())]), # A pipeline that first creates polynomial features (e.g., squares, cubes) and then applies linear regression. This can capture non‑linear patterns.
    'SGDRegressor': make_pipeline(SGDRegressor()), # Linear model trained using Stochastic Gradient Descent, good for large datasets.
    'ANN': make_pipeline(
        MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    ), # A simple neural network with one hidden layer of 100 neurons. It can learn complex patterns.
    'RandomForest': RandomForestRegressor(), #An ensemble of many decision trees. It averages their predictions to improve accuracy and reduce overfitting.
    'SVM': make_pipeline(SVR()), #Uses a kernel to map data into higher dimensions and find a hyperplane that fits the data.
    'LGBM': lgb.LGBMRegressor(), # A fast, gradient‑boosted decision tree algorithm.
    'XGBoost': xgb.XGBRegressor(), # Another popular gradient boosting method, often very accurate.
    'KNN': make_pipeline(KNeighborsRegressor()) # Predicts the price by averaging the prices of the k most similar houses in the training set.
    }

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)        # Train
    y_pred = model.predict(X_test)     # Predict on test set
    
# Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
        # Save results
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
    
        # Save the trained model to a file
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

#After training, it calculates three common regression metrics:

# MAE (Mean Absolute Error) – Average absolute difference between predicted and actual prices. Lower is better.

# MSE (Mean Squared Error) – Average squared difference. Punishes large errors more.

# R² (R-squared) – Proportion of variance in the target explained by the model. Ranges from 0 to 1 (closer to 1 is better).

# Finally, it saves the trained models as .pkl files (pickle) so they can be reused later without retraining.


# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")
