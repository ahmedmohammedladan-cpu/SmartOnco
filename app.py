from flask import Flask, render_template, request
import joblib, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
import matplotlib.pyplot as plt
import io, base64
import os

app = Flask(__name__)

# ==================================================
# BREAST CANCER SECTION - SIMPLE FIX
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

print("🔧 Initializing Breast Cancer Model with FIX...")
data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target  # 0 = malignant, 1 = benign

print(f"Dataset: Malignant (0): {sum(y_bc == 0)}, Benign (1): {sum(y_bc == 1)}")

# SIMPLE FIX: Adjust the model to be MORE sensitive to malignancies
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

# FIXED MODEL: Decision Tree with MALIGNANT bias
model_bc = DecisionTreeClassifier(
    random_state=42,
    class_weight={0: 5, 1: 1},  # ⭐⭐⭐ CRITICAL: Malignant gets 5x weight ⭐⭐⭐
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='entropy'
)

model_bc.fit(X_train_bc, y_train_bc)
joblib.dump(model_bc, "breast_cancer_model_fixed.pkl")

demo_values_bc = X_test_bc[0].tolist()
print("✅ Breast Cancer Model Ready with MALIGNANT bias")

# ==================================================
# LUNG CANCER SECTION
# ==================================================
try:
    data_lc = pd.read_csv("survey lung cancer.csv")
    if "GENDER" in data_lc.columns:
        data_lc["GENDER"] = data_lc["GENDER"].map({"M": 1, "F": 0})
    if "LUNG_CANCER" in data_lc.columns:
        data_lc["LUNG_CANCER"] = data_lc["LUNG_CANCER"].map({"NO": 0, "YES": 1})

    if "LUNG_CANCER" in data_lc.columns:
        X_lc = data_lc.drop("LUNG_CANCER", axis=1)
        y_lc = data_lc["LUNG_CANCER"]
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
            X_lc, y_lc, test_size=0.2, random_state=42, stratify=y_lc
        )
        
        model_lc = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=6
        )
        model_lc.fit(X_train_lc, y_train_lc)
        joblib.dump(model_lc, "lung_cancer_model.pkl")
        
        demo_values_lc = X_test_lc.iloc[0].tolist()
        selected_features_lc = list(X_lc.columns)
        print("✅ Lung Cancer Model Ready")
        
except Exception as e:
    print(f"⚠️ Lung Cancer Model Error: {e}")
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
# SIMPLE PREDICTION FUNCTION
# ==================================================
def get_prediction_text(pred, proba, cancer_type="breast"):
    """Simple prediction formatting"""
    if cancer_type == "breast":
        # In breast cancer: 0 = malignant, 1 = benign
        malignant_prob = proba[0] * 100
        benign_prob = proba[1] * 100
        
        if pred == 0:  # Malignant
            if malignant_prob >= 80:
                risk = "High risk"
            elif malignant_prob >= 60:
                risk = "Moderate risk"
            elif malignant_prob >= 40:
                risk = "Low risk"
            else:
                risk = "Very low risk"
            return f"Prediction: Malignant (Cancerous) — {risk} ({malignant_prob:.1f}% malignant probability)"
        else:  # Benign
            if benign_prob >= 80:
                risk = "Very low risk"
            elif benign_prob >= 60:
                risk = "Low risk"
            else:
                risk = "Monitor"
            return f"Prediction: Benign (Non-cancerous) — {risk} ({benign_prob:.1f}% benign probability)"
    
    elif cancer_type == "lung":
        # In lung cancer: 0 = no cancer, 1 = cancer
        cancer_prob = proba[1] * 100
        
        if pred == 1:  # Cancer
            if cancer_prob >= 80:
                risk = "High risk"
            elif cancer_prob >= 60:
                risk = "Moderate risk"
            elif cancer_prob >= 40:
                risk = "Low risk"
            else:
                risk = "Very low risk"
            return f"Prediction: Lung Cancer Detected — {risk} ({cancer_prob:.1f}% probability)"
        else:  # No cancer
            no_cancer_prob = proba[0] * 100
            if no_cancer_prob >= 80:
                risk = "Very low risk"
            elif no_cancer_prob >= 60:
                risk = "Low risk"
            else:
                risk = "Monitor"
            return f"Prediction: No Lung Cancer — {risk} ({no_cancer_prob:.1f}% probability)"

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc)

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
        # Get features
        feature_vals = [float(request.form.get(f"feature_bc{i+1}")) for i in range(len(selected_features_bc))]
        df = pd.DataFrame([feature_vals], columns=selected_features_bc)
        
        # Get prediction
        pred = model_bc.predict(df)[0]
        proba = model_bc.predict_proba(df)[0]
        
        # Get prediction text
        prediction_text = get_prediction_text(pred, proba, "breast")
        
        # DEBUG: Print probabilities
        print(f"DEBUG - Malignant prob: {proba[0]*100:.1f}%, Benign prob: {proba[1]*100:.1f}%")
        
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

# ==================================================
# QUICK TEST PAGE
# ==================================================
@app.route("/quick_test")
def quick_test():
    """Quick test of the problematic cases"""
    
    # Test the problematic cases
    test_cases = [
        ("Case 3 (Borderline)", [14.99, 0.02701, 97.65, 0.1471, 0.0701, 14.25, 450.9, 94.96, 17.94, 0.1015]),
        ("Case 5 (Borderline)", [16.25, 0.03735, 108.4, 0.1822, 0.0953, 14.92, 1022.0, 97.65, 19.61, 0.1099]),
        ("Clear Malignant", [20.57, 0.05185, 132.9, 0.2788, 0.1615, 17.99, 1326.0, 122.8, 20.38, 0.1186]),
        ("Clear Benign", [13.54, 0.01335, 87.46, 0.0456, 0.0380, 13.08, 566.3, 85.63, 15.71, 0.09797])
    ]
    
    results = []
    for name, features in test_cases:
        df = pd.DataFrame([features], columns=selected_features_bc)
        pred = model_bc.predict(df)[0]
        proba = model_bc.predict_proba(df)[0]
        
        results.append({
            "name": name,
            "malignant_prob": f"{proba[0]*100:.1f}%",
            "benign_prob": f"{proba[1]*100:.1f}%",
            "prediction": "MALIGNANT" if pred == 0 else "BENIGN",
            "expected": "MALIGNANT" if name in ["Case 3", "Case 5", "Clear Malignant"] else "BENIGN"
        })
    
    # Generate HTML
    html = """
    <html>
    <head><title>Quick Test - Borderline Cases</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 12px; }
        th { background: #f4f4f4; }
        .correct { background: #d4ffd4; }
        .wrong { background: #ffd4d4; }
        h2 { color: #333; }
    </style>
    </head>
    <body>
        <h2>🧪 Quick Test - Previously Failing Cases</h2>
        <p>Testing if borderline malignant cases are now detected</p>
        
        <table>
            <tr><th>Case</th><th>Malignant Prob</th><th>Benign Prob</th><th>Prediction</th><th>Expected</th><th>Status</th></tr>
    """
    
    for r in results:
        correct = r["prediction"] == r["expected"]
        status = "✅ PASS" if correct else "❌ FAIL"
        row_class = "correct" if correct else "wrong"
        
        html += f"""
            <tr class="{row_class}">
                <td><strong>{r['name']}</strong></td>
                <td>{r['malignant_prob']}</td>
                <td>{r['benign_prob']}</td>
                <td>{r['prediction']}</td>
                <td>{r['expected']}</td>
                <td>{status}</td>
            </tr>
        """
    
    # Check if fixed
    case3_fixed = results[0]["prediction"] == "MALIGNANT" and float(results[0]["malignant_prob"].replace('%', '')) > 50
    case5_fixed = results[1]["prediction"] == "MALIGNANT" and float(results[1]["malignant_prob"].replace('%', '')) > 50
    
    html += f"""
        </table>
        
        <div style="margin-top: 30px; padding: 20px; background: {'#d4ffd4' if case3_fixed and case5_fixed else '#ffd4d4'}; border-radius: 5px;">
            <h3>FIX STATUS: {'✅ FIXED!' if case3_fixed and case5_fixed else '❌ STILL BROKEN'}</h3>
            <p><strong>Case 3 (Borderline):</strong> {results[0]['malignant_prob']} malignant - {'✅ NOW MALIGNANT' if case3_fixed else '❌ Still benign'}</p>
            <p><strong>Case 5 (Borderline):</strong> {results[1]['malignant_prob']} malignant - {'✅ NOW MALIGNANT' if case5_fixed else '❌ Still benign'}</p>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Test in Main System:</h3>
            <a href="/" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">
                Go to Breast Cancer Diagnosis
            </a>
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
    print("🚀 SMARTONCO - SIMPLE FIX VERSION")
    print("="*60)
    print("Class weights: Malignant(0)=5x, Benign(1)=1x")
    print("Test borderline cases: http://localhost:5000/quick_test")
    print("="*60)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
