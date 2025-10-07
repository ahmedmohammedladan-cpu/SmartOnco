import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("survey lung cancer.csv")

# Encode categorical columns
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model_lc = DecisionTreeClassifier(random_state=42)
model_lc.fit(X_train, y_train)

# Save model
joblib.dump(model_lc, "lung_cancer_model.pkl")

print("✅ Lung cancer model trained and saved!")
