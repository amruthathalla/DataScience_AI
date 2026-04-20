import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Boosting Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ================================
# Load Dataset
# ================================
dataset = pd.read_csv(r"C:/Users/Amrutha Thalla/FSDS/DataScience_AI/NLP - AI/nlp_project/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)


# ================================
# Data Expansion Function
# ================================
def expand_dataset(X, y, times=2):
    X_expanded = np.vstack([X] * times)
    y_expanded = np.hstack([y] * times)

    # Shuffle after expansion
    from sklearn.utils import shuffle
    X_expanded, y_expanded = shuffle(X_expanded, y_expanded, random_state=0)

    return X_expanded, y_expanded

# ================================
# Text Cleaning Function
# ================================
ps = PorterStemmer()

def clean_text(use_stopwords=True):
    corpus = []

    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower().split()

        if use_stopwords:
            review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        else:
            review = [ps.stem(word) for word in review]

        review = ' '.join(review)
        corpus.append(review)

    return corpus

# ================================
# Prepare Both Versions
# ================================
corpus_with_sw = clean_text(use_stopwords=True)
corpus_without_sw = clean_text(use_stopwords=False)

y = dataset.iloc[:, 1].values

# ================================
# Feature Extraction Function
# ================================
def get_features(corpus, method="tfidf"):
    if method == "bow":
        vectorizer = CountVectorizer(max_features=2000)
    else:
        vectorizer = TfidfVectorizer(max_features=2000)

    X = vectorizer.fit_transform(corpus).toarray()
    return X

# ================================
# Model Runner
# ================================
def run_models(X, label):

    print(f"\n=========== {label} ===========\n")

    # EXPAND DATA (2x or 3x)
    X, y_expanded = expand_dataset(X, y, times=3)   # try 2 or 3

    # Train-test split AFTER expansion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_expanded, test_size=0.20, random_state=0
    )

    models = {
        "Logistic": LogisticRegression(max_iter=200),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            bias = model.score(X_train, y_train)
            variance = model.score(X_test, y_test)

            print(f"Model: {name}")
            print("Accuracy:", acc)
            print("Bias:", bias)
            print("Variance:", variance)
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("-" * 50)

        except Exception as e:
            print(f"{name} failed due to: {e}")
            print("-" * 50)
# ================================
# Run All Experiments
# ================================

# 1. TF-IDF with Stopwords removed
X1 = get_features(corpus_with_sw, "tfidf")
run_models(X1, "TF-IDF WITH STOPWORDS REMOVED")

# 2. TF-IDF without removing stopwords
X2 = get_features(corpus_without_sw, "tfidf")
run_models(X2, "TF-IDF WITHOUT STOPWORDS REMOVED")

# 3. BoW with Stopwords removed
X3 = get_features(corpus_with_sw, "bow")
run_models(X3, "BOW WITH STOPWORDS REMOVED")

# 4. BoW without removing stopwords
X4 = get_features(corpus_without_sw, "bow")
run_models(X4, "BOW WITHOUT STOPWORDS REMOVED")