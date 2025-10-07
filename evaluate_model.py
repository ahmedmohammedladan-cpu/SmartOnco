# evaluate_model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load the saved model
model = joblib.load("decision_tree_model.pkl")

# 2) Load the dataset and make a test split (same as training code)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3) Predict on the held-out test set
y_pred = model.predict(X_test)

# 4) Report metrics
print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
print("\nConfusion Matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))

# 5) Show a few sample predictions
print("\nSample predictions (first 5 rows):")
for i in range(5):
    print(f"True: {data.target_names[y_test.iloc[i]]:9s}  Pred: {data.target_names[y_pred[i]]}")
