import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== STEP 1: LOAD DATA ==========
try:
    data = pd.read_csv("prostate_cancer.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Dataset 'prostate_cancer.csv' not found in this folder.")
    exit()

print("\n📊 First 5 rows:")
print(data.head())

# ========== STEP 2: CLEAN / ENCODE ==========
# Try to detect target column automatically
target_candidates = ["diagnosis", "class", "target", "label", "cancer", "result"]
target_col = None

for col in data.columns:
    if col.lower() in target_candidates:
        target_col = col
        break

if target_col is None:
    # fallback: assume last column is the target
    target_col = data.columns[-1]
    print(f"\n⚠️ Couldn't detect target automatically. Using last column: '{target_col}'")

# Encode non-numeric columns if needed
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype("category").cat.codes

# ========== STEP 3: SPLIT DATA ==========
X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== STEP 4: TRAIN MODEL ==========
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ========== STEP 5: EVALUATE ==========
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained successfully with accuracy: {acc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ========== STEP 6: SAVE MODEL ==========
joblib.dump(model, "prostate_cancer_model.pkl")
print("\n💾 Model saved as 'prostate_cancer_model.pkl'")
