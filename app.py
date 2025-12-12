from flask import Flask, render_template, request
import joblib, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io, base64
import os
from datetime import datetime

app = Flask(__name__)

# ==================================================
# BREAST CANCER SECTION - FINAL FIX!
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

print("🔧 Initializing Breast Cancer Model...")
data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target  # IMPORTANT: In Wisconsin dataset: 0 = malignant, 1 = benign

print(f"Data shape: {X_bc.shape}")
print(f"ORIGINAL: Malignant (0): {sum(y_bc == 0)}, Benign (1): {sum(y_bc == 1)}")

# CRITICAL FIX: NO LABEL REVERSAL NEEDED!
# The model will learn that 0 = malignant, 1 = benign
# We'll just interpret probabilities correctly in prediction function

# Feature enhancement
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)

# Use 4 clusters to better capture patterns
kmeans_bc = KMeans(n_clusters=4, random_state=42, n_init=20)
kmeans_labels_bc = kmeans_bc.fit_predict(X_scaled_bc)

# Add cluster distances as features
distances_bc = kmeans_bc.transform(X_scaled_bc)
X_enhanced_bc = np.hstack([X_bc, distances_bc])

# Split data
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_enhanced_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

print(f"Training: Malignant (0): {sum(y_train_bc == 0)}, Benign (1): {sum(y_train_bc == 1)}")

# Train model - Give more weight to MALIGNANT (class 0)
model_bc = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight={0: 3, 1: 1},  # Malignant (0) has 3x weight of Benign (1)
    random_state=42
)

print("Training model with malignant bias...")
model_bc.fit(X_train_bc, y_train_bc)

# Calibrate probabilities
calibrator_bc = CalibratedClassifierCV(model_bc, cv=5, method='isotonic')
calibrator_bc.fit(X_train_bc, y_train_bc)

# Save model
joblib.dump({
    'model': calibrator_bc,
    'scaler': scaler_bc,
    'kmeans': kmeans_bc,
    'feature_names': selected_features_bc,
    'class_names': ['Malignant', 'Benign']  # 0=Malignant, 1=Benign
}, "breast_cancer_model_fixed.pkl")

demo_values_bc = X_test_bc[0, :10].tolist()
print("✅ Breast Cancer Model Initialized - Malignant=0, Benign=1")

# ==================================================
# LUNG CANCER SECTION
# ==================================================
print("\n🔧 Initializing Lung Cancer Model...")
try:
    data_lc = pd.read_csv("survey lung cancer.csv")
    if "GENDER" in data_lc.columns:
        data_lc["GENDER"] = data_lc["GENDER"].map({"M": 1, "F": 0})
    if "LUNG_CANCER" in data_lc.columns:
        data_lc["LUNG_CANCER"] = data_lc["LUNG_CANCER"].map({"NO": 0, "YES": 1})

    if "LUNG_CANCER" in data_lc.columns:
        X_lc = data_lc.drop("LUNG_CANCER", axis=1)
        y_lc = data_lc["LUNG_CANCER"]
        
        print(f"Lung - Cancer (1): {sum(y_lc == 1)}, No cancer (0): {sum(y_lc == 0)}")
        
        # Process lung data
        scaler_lc = StandardScaler()
        X_scaled_lc = scaler_lc.fit_transform(X_lc)
        
        kmeans_lc = KMeans(n_clusters=3, random_state=42, n_init=20)
        kmeans_lc.fit(X_scaled_lc)
        
        distances_lc = kmeans_lc.transform(X_scaled_lc)
        X_enhanced_lc = np.hstack([X_lc.values, distances_lc])
        
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
            X_enhanced_lc, y_lc, test_size=0.2, random_state=42, stratify=y_lc
        )
        
        model_lc = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        calibrator_lc = CalibratedClassifierCV(model_lc, cv=5, method='sigmoid')
        calibrator_lc.fit(X_train_lc, y_train_lc)
        model_lc = calibrator_lc
        
        joblib.dump(model_lc, "lung_cancer_model.pkl")
        demo_values_lc = X_test_lc[0, :X_lc.shape[1]].tolist()
        selected_features_lc = list(X_lc.columns) + [f'Cluster_Dist_{i}' for i in range(3)]
        
        print("✅ Lung Cancer Model Initialized")
        
except Exception as e:
    print(f"⚠️ Lung Cancer Model Error: {e}")
    data_lc = None
    model_lc = None
    demo_values_lc = [1, 62, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    selected_features_lc = [
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
        "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
        "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
        "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
    ]

# ==================================================
# PROSTATE CANCER SECTION
# ==================================================
features_prostate = [
    "Age", "PSA_Level", "Biopsy_Result", "Tumor_Size", "Cancer_Stage",
    "Blood_Pressure", "Cholesterol_Level", "Family_History", 
    "Smoking_History", "Alcohol_Consumption", "Back_Pain", "Fatigue_Level"
]

demo_values_prostate = {
    "Age": 65,
    "PSA_Level": 8.5,
    "Biopsy_Result": 1,
    "Tumor_Size": 2.5,
    "Cancer_Stage": 3,
    "Blood_Pressure": 145,
    "Cholesterol_Level": 240,
    "Family_History": 1,
    "Smoking_History": 1,
    "Alcohol_Consumption": 1,
    "Back_Pain": 1,
    "Fatigue_Level": 1
}

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_risk_category(probability, is_cancer=True):
    """Consistent risk categorization"""
    if is_cancer:
        if probability >= 0.85:
            return "High risk", probability * 100
        elif probability >= 0.60:
            return "Moderate risk", probability * 100
        elif probability >= 0.30:
            return "Low risk", probability * 100
        else:
            return "Very low risk", probability * 100
    else:
        benign_prob = 1 - probability
        if benign_prob >= 0.85:
            return "Very low risk", benign_prob * 100
        elif benign_prob >= 0.60:
            return "Low risk", benign_prob * 100
        else:
            return "Monitor", benign_prob * 100

def prepare_breast_features(feature_vals):
    """Prepare breast cancer features with cluster enhancement"""
    try:
        scaled_vals = scaler_bc.transform([feature_vals])
        distances = kmeans_bc.transform(scaled_vals)
        enhanced_vals = np.hstack([feature_vals, distances[0]])
        return enhanced_vals
    except:
        return feature_vals

# ==================================================
# ROUTES - FINAL FIXED VERSION
# ==================================================
@app.route("/")
def home():
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc)

# ---------------- BREAST CANCER - FINAL FIX ----------------
@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
        # Get feature values
        feature_vals = [float(request.form.get(f"feature_bc{i+1}")) for i in range(len(selected_features_bc))]
        
        # Prepare enhanced features
        enhanced_features = prepare_breast_features(feature_vals)
        
        # Get probability of MALIGNANT (class 0) and BENIGN (class 1)
        probabilities = calibrator_bc.predict_proba([enhanced_features])[0]
        malignant_prob = probabilities[0]  # Class 0 = Malignant
        benign_prob = probabilities[1]     # Class 1 = Benign
        
        print(f"DEBUG: Malignant prob (class 0) = {malignant_prob:.3f}, Benign prob (class 1) = {benign_prob:.3f}")
        
        # Determine prediction - Malignant if malignant_prob > benign_prob
        if malignant_prob > benign_prob:
            label = "Malignant (Cancerous)"
            risk_level, confidence = get_risk_category(malignant_prob, is_cancer=True)
            # Show MALIGNANT probability
            prediction_text = f"Prediction: {label} — {risk_level} ({malignant_prob*100:.1f}% malignant probability)"
        else:
            label = "Benign (Non-cancerous)"
            risk_level, confidence = get_risk_category(malignant_prob, is_cancer=False)
            # Show BENIGN probability
            prediction_text = f"Prediction: {label} — {risk_level} ({benign_prob*100:.1f}% benign probability)"
        
        return render_template(
            "index.html", 
            features_bc=selected_features_bc, 
            demo_values_bc=demo_values_bc, 
            prediction_text=prediction_text
        )
    except Exception as e:
        return render_template(
            "index.html", 
            features_bc=selected_features_bc, 
            demo_values_bc=demo_values_bc, 
            prediction_text=f"Error: {str(e)}"
        )

# ---------------- TEST ENDPOINT FOR BORDERLINE CASES ----------------
@app.route("/test_fix", methods=["GET"])
def test_fix():
    """Test the previously failing borderline cases"""
    
    # Test cases that previously failed
    test_cases = [
        {
            "name": "Case 3 - Borderline Malignant (PREVIOUSLY FAILED)",
            "features": [14.99, 0.02701, 97.65, 0.1471, 0.0701, 14.25, 450.9, 94.96, 17.94, 0.1015],
            "expected": "MALIGNANT"
        },
        {
            "name": "Case 5 - Borderline Malignant (PREVIOUSLY FAILED)",
            "features": [16.25, 0.03735, 108.4, 0.1822, 0.0953, 14.92, 1022.0, 97.65, 19.61, 0.1099],
            "expected": "MALIGNANT"
        },
        {
            "name": "Clear Malignant (Case 1)",
            "features": [20.57, 0.05185, 132.9, 0.2788, 0.1615, 17.99, 1326.0, 122.8, 20.38, 0.1186],
            "expected": "MALIGNANT"
        },
        {
            "name": "Clear Benign (Case 2)",
            "features": [13.54, 0.01335, 87.46, 0.0456, 0.0380, 13.08, 566.3, 85.63, 15.71, 0.09797],
            "expected": "BENIGN"
        }
    ]
    
    results = []
    for case in test_cases:
        enhanced = prepare_breast_features(case["features"])
        probabilities = calibrator_bc.predict_proba([enhanced])[0]
        malignant_prob = probabilities[0] * 100
        benign_prob = probabilities[1] * 100
        
        prediction = "MALIGNANT" if malignant_prob > benign_prob else "BENIGN"
        correct = prediction == case["expected"]
        
        results.append({
            "case": case["name"],
            "malignant_prob": f"{malignant_prob:.1f}%",
            "benign_prob": f"{benign_prob:.1f}%",
            "prediction": prediction,
            "expected": case["expected"],
            "status": "✅ CORRECT" if correct else "❌ WRONG"
        })
    
    # Generate HTML report
    html = """
    <html>
    <head>
        <title>SmartOnco Fix Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f4f4f4; }
            .correct { background-color: #d4edda; }
            .wrong { background-color: #f8d7da; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>🧪 SmartOnco Borderline Case Fix Test</h1>
        <p>Testing previously failing malignant cases</p>
        
        <table>
            <tr>
                <th>Test Case</th>
                <th>Malignant Prob</th>
                <th>Benign Prob</th>
                <th>Prediction</th>
                <th>Expected</th>
                <th>Status</th>
            </tr>
    """
    
    for r in results:
        row_class = "correct" if r["status"] == "✅ CORRECT" else "wrong"
        html += f"""
            <tr class="{row_class}">
                <td><strong>{r['case']}</strong></td>
                <td>{r['malignant_prob']}</td>
                <td>{r['benign_prob']}</td>
                <td>{r['prediction']}</td>
                <td>{r['expected']}</td>
                <td>{r['status']}</td>
            </tr>
        """
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r["status"] == "✅ CORRECT")
    accuracy = (correct_count / len(results)) * 100
    
    html += f"""
        </table>
        
        <div style="margin-top: 30px; padding: 20px; background-color: {'#d4edda' if accuracy >= 75 else '#f8d7da'}; border-radius: 5px;">
            <h3>Overall Results: {correct_count}/{len(results)} correct ({accuracy:.1f}%)</h3>
            <p><strong>Key Test:</strong> Cases 3 & 5 must show MALIGNANT with probability > 50%</p>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Quick Manual Test:</h3>
            <form action="/predict_bc" method="POST" style="background: #f9f9f9; padding: 20px; border-radius: 5px;">
                <h4>Test Case 3 (Borderline Malignant):</h4>
    """
    
    # Add form for testing Case 3
    case3_features = test_cases[0]["features"]
    for i, (feature_name, value) in enumerate(zip(selected_features_bc, case3_features), 1):
        html += f"""
                <label>{feature_name}:</label>
                <input type="number" step="0.00001" name="feature_bc{i}" value="{value}" style="width: 100px; margin: 5px;">
                <br>
        """
    
    html += """
                <br>
                <input type="submit" value="Test This Case" style="padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
            </form>
        </div>
    </body>
    </html>
    """
    
    return html

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SMARTONCO SYSTEM - FINAL FIXED VERSION")
    print("="*60)
    print("IMPORTANT: Malignant = Class 0, Benign = Class 1")
    print("Test borderline cases at: http://localhost:5000/test_fix")
    print("="*60)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
