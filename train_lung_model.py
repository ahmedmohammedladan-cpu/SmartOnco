import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
data = pd.read_csv("survey lung cancer.csv")

# 2. Preprocess
# Encode categorical values
data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})
data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# Features (all except target)
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Train Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save Model
joblib.dump(model, "lung_cancer_model.pkl")
print("\n💾 Model saved as lung_cancer_model.pkl")
