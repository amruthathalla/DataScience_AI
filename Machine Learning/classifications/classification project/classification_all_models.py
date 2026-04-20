# =============================================================================
# COMPLETE BACKEND: Train 9 ML Models and Save as Pickle Files
# 
# WHAT THIS SCRIPT DOES:
#   1. Loads a CSV file (e.g., Social_Network_Ads.csv)
#   2. Splits data into train/test (80%/20%)
#   3. Scales features using StandardScaler
#   4. Trains 9 classification algorithms
#   5. Saves each trained model as a .pkl file
#   6. Saves the scaler as scaler.pkl
#   7. Prints performance summary and saves results.csv
#
# HOW TO RUN:
#   python backend_train_and_save_models.py
#
# REQUIRED LIBRARIES:
#   pip install pandas numpy scikit-learn xgboost lightgbm
# =============================================================================

# ----------------------------- 1. IMPORTS ------------------------------------
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Import all classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB          # works with continuous features
from sklearn.neural_network import MLPClassifier


# ----------------------------- 2. CONFIGURATION ------------------------------
# 👉 CHANGE THIS PATH TO YOUR CSV FILE
df = r"file:///C:/Users/Amrutha Thalla/FSDS/DataScience_AI/Machine Learning/classifications/classification project/logit classification.csv"
# Output folder for saved models
OUTPUT_DIR = r"C:/Users/Amrutha Thalla/FSDS/DataScience_AI/Machine Learning/classifications/classification project/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test split ratio (20% for testing)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ----------------------------- 3. LOAD DATA ----------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

try:
    df = pd.read_csv(df)
    print(f"✅ Loaded: {df}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"❌ File not found: {df}")
    print("Please update Df to your CSV file.")
    exit(1)

# ----------------------------- 4. PREPARE DATA -------------------------------
print("\n" + "=" * 60)
print("STEP 2: PREPARING DATA (Features & Target)")
print("=" * 60)

# Assume the last column is the target (Purchased)
# Features: Age (col 2) and EstimatedSalary (col 3) – adjust if needed
X = df.iloc[:, [2, 3]].values   # Age and Salary
y = df.iloc[:, -1].values       # Target (last column)

print(f"Features shape: {X.shape} (Age, Salary)")
print(f"Target shape:   {y.shape}")
print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# Feature scaling (very important for SVM, KNN, ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Feature scaling completed (StandardScaler)")

# ----------------------------- 5. DEFINE MODELS ------------------------------
print("\n" + "=" * 60)
print("STEP 3: DEFINING MODELS")
print("=" * 60)

models = {}

# Core models (always available)
models["Logistic Regression"] = LogisticRegression(random_state=RANDOM_STATE)
models["SVM"] = SVC(probability=True, random_state=RANDOM_STATE)
models["KNN"] = KNeighborsClassifier(n_neighbors=5, p=2)
models["Decision Tree"] = DecisionTreeClassifier(random_state=RANDOM_STATE)
models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
models["Gaussian NB"] = GaussianNB()
models["ANN"] = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                              max_iter=500, random_state=RANDOM_STATE)

# Optional advanced models
if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(n_estimators=100, use_label_encoder=False,
                                      eval_metric='logloss', random_state=RANDOM_STATE)
if LGBM_AVAILABLE:
    models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)

print(f"Total models to train: {len(models)}")
for name in models.keys():
    print(f"  - {name}")

# ----------------------------- 6. TRAIN & SAVE MODELS ------------------------
print("\n" + "=" * 60)
print("STEP 4: TRAINING MODELS & SAVING PICKLES")
print("=" * 60)

results = []

for name, model in models.items():
    print(f"\nTraining: {name} ...", end=" ", flush=True)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Save model as pickle file
    model_filename = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    
    # Save results for summary
    results.append({
        "Model": name,
        "Test Accuracy (%)": round(acc * 100, 2),
        "Saved Path": model_filename
    })
    
    print(f"Done. Accuracy = {acc*100:.2f}% → Saved to {model_filename}")

# Save the scaler separately (needed for preprocessing new data)
scaler_filename = os.path.join(OUTPUT_DIR, "scaler.pkl")
with open(scaler_filename, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n✅ Scaler saved to: {scaler_filename}")

# ----------------------------- 7. SUMMARY & RESULTS --------------------------
print("\n" + "=" * 60)
print("STEP 5: PERFORMANCE SUMMARY")
print("=" * 60)

df_results = pd.DataFrame(results).sort_values("Test Accuracy (%)", ascending=False)
df_results.index = range(1, len(df_results) + 1)
print(df_results.to_string())

# Save results to CSV
results_csv = os.path.join(OUTPUT_DIR, "model_performance.csv")
df_results.to_csv(results_csv, index=False)
print(f"\n📊 Performance summary saved to: {results_csv}")

# ----------------------------- 8. DETAILS OF BEST MODEL ---------------------
best_model_name = df_results.iloc[0]["Model"]
best_model = models[best_model_name]

print("\n" + "=" * 60)
print(f"🏆 BEST MODEL: {best_model_name}")
print("=" * 60)

# Predict on test set with best model
y_pred_best = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print("            Predicted")
print("             No   Yes")
print(f"Actual No   {cm[0,0]:3}  {cm[0,1]:3}")
print(f"       Yes  {cm[1,0]:3}  {cm[1,1]:3}")

# ----------------------------- 9. EXAMPLE LOAD & PREDICT --------------------
print("\n" + "=" * 60)
print("STEP 6: TESTING SAVED MODEL (Load & Predict)")
print("=" * 60)

# Example: load the best model and scaler, then predict on a new customer
with open(model_filename, "rb") as f:
    loaded_model = pickle.load(f)

with open(scaler_filename, "rb") as f:
    loaded_scaler = pickle.load(f)

# New customer: Age = 35, Salary = 75000
new_customer = np.array([[35, 75000]])
new_customer_scaled = loaded_scaler.transform(new_customer)
prediction = loaded_model.predict(new_customer_scaled)[0]

print(f"New customer: Age 35, Salary 75000")
print(f"Prediction: {'Will Purchase' if prediction == 1 else 'Will NOT Purchase'}")

if hasattr(loaded_model, "predict_proba"):
    prob = loaded_model.predict_proba(new_customer_scaled)[0][1]
    print(f"Confidence: {prob*100:.1f}%")

# ----------------------------- 10. DONE -------------------------------------
print("\n" + "=" * 60)
print("✅ ALL DONE!")
print(f"📁 Models saved in folder: {OUTPUT_DIR}/")
print("   You can now use these pickle files in your Streamlit frontend.")
print("=" * 60)