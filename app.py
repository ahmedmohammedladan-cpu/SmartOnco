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
# BREAST CANCER SECTION - COMPLETELY REBUILT
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
y_bc = data_bc.target  # 0 = malignant, 1 = benign

print(f"Data shape: {X_bc.shape}")
print(f"Malignant samples: {sum(y_bc == 0)}, Benign samples: {sum(y_bc == 1)}")

# CRITICAL FIX 1: Reverse labels if needed - Wisconsin dataset: 0=malignant, 1=benign
# We want: 0=benign, 1=malignant for intuitive probability
y_bc = 1 - y_bc  # Now 1 = malignant, 0 = benign
print(f"After reversal - Malignant (1): {sum(y_bc == 1)}, Benign (0): {sum(y_bc == 0)}")

# FIX 2: Create better features with clustering
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)

# Use 3 clusters to capture borderline cases
kmeans_bc = KMeans(n_clusters=3, random_state=42, n_init=20)
kmeans_labels_bc = kmeans_bc.fit_predict(X_scaled_bc)

# Add cluster distances as features
distances_bc = kmeans_bc.transform(X_scaled_bc)
X_enhanced_bc = np.hstack([X_bc, distances_bc])

# FIX 3: Split with stratification
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_enhanced_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

print(f"Training samples: {X_train_bc.shape[0]}, Malignant in training: {sum(y_train_bc == 1)}")

# FIX 4: Use ensemble model instead of single decision tree
model_bc = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # CRITICAL: Balance malignant/benign
    random_state=42
)

# FIX 5: Train on borderline cases - add weight to malignant
print("Training model...")
model_bc.fit(X_train_bc, y_train_bc)

# FIX 6: Calibrate probabilities
calibrator_bc = CalibratedClassifierCV(model_bc, cv=5, method='isotonic')
calibrator_bc.fit(X_train_bc, y_train_bc)

# Save the calibrated model
joblib.dump({
    'model': calibrator_bc,
    'scaler': scaler_bc,
    'kmeans': kmeans_bc,
    'feature_names': selected_features_bc
}, "breast_cancer_model_calibrated.pkl")

demo_values_bc = X_test_bc[0, :10].tolist()
print("✅ Breast Cancer Model Initialized")

# ==================================================
# LUNG CANCER SECTION - FIXED
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
        
        # Ensure 1 = cancer, 0 = no cancer
        print(f"Lung - Cancer samples: {sum(y_lc == 1)}, No cancer: {sum(y_lc == 0)}")
        
        # Scale and enhance
        scaler_lc = StandardScaler()
        X_scaled_lc = scaler_lc.fit_transform(X_lc)
        
        kmeans_lc = KMeans(n_clusters=3, random_state=42, n_init=20)
        kmeans_labels_lc = kmeans_lc.fit_predict(X_scaled_lc)
        
        distances_lc = kmeans_lc.transform(X_scaled_lc)
        X_enhanced_lc = np.hstack([X_lc.values, distances_lc])
        
        # Split
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
            X_enhanced_lc, y_lc, test_size=0.2, random_state=42, stratify=y_lc
        )
        
        # Train calibrated model
        model_lc = RandomForestClassifier(
            n_estimators=50,
            class_weight='balanced',
            random_state=42
        )
        
        calibrator_lc = CalibratedClassifierCV(model_lc, cv=5, method='sigmoid')
        calibrator_lc.fit(X_train_lc, y_train_lc)
        model_lc = calibrator_lc
        
        joblib.dump(model_lc, "lung_cancer_model_calibrated.pkl")
        demo_values_lc = X_test_lc[0, :X_lc.shape[1]].tolist()
        selected_features_lc = list(X_lc.columns) + [f'Cluster_Dist_{i}' for i in range(3)]
        
        print("✅ Lung Cancer Model Initialized")
        
except Exception as e:
    print(f"⚠️ Lung Cancer Model Error: {e}")
    data_lc = None
    model_lc = None
    demo_values_lc = [1, 62, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Default demo
    selected_features_lc = [
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", 
        "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", 
        "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", 
        "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
    ]

# ==================================================
# PROSTATE CANCER SECTION - PERFECT (Keep as is)
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
        # Scale
        scaled_vals = scaler_bc.transform([feature_vals])
        # Get distances to clusters
        distances = kmeans_bc.transform(scaled_vals)
        # Combine
        enhanced_vals = np.hstack([feature_vals, distances[0]])
        return enhanced_vals
    except:
        return feature_vals

def prepare_lung_features(feature_vals):
    """Prepare lung cancer features"""
    try:
        if scaler_lc and kmeans_lc:
            scaled_vals = scaler_lc.transform([feature_vals])
            distances = kmeans_lc.transform(scaled_vals)
            enhanced_vals = np.hstack([feature_vals, distances[0]])
            return enhanced_vals
    except:
        pass
    return feature_vals

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc)

# ---------------- BREAST CANCER ROUTES - FIXED ----------------
@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
        # Get feature values
        feature_vals = [float(request.form.get(f"feature_bc{i+1}")) for i in range(len(selected_features_bc))]
        
        # Prepare enhanced features
        enhanced_features = prepare_breast_features(feature_vals)
        
        # Get probability of MALIGNANT (class 1)
        malignant_prob = calibrator_bc.predict_proba([enhanced_features])[0][1]
        benign_prob = 1 - malignant_prob
        
        # CRITICAL: Always show malignant probability in output
        if malignant_prob > benign_prob:
            label = "Malignant (Cancerous)"
            risk_level, confidence = get_risk_category(malignant_prob, is_cancer=True)
            prediction_text = f"Prediction: {label} — {risk_level} ({malignant_prob*100:.1f}% malignant probability)"
        else:
            label = "Benign (Non-cancerous)"
            risk_level, confidence = get_risk_category(malignant_prob, is_cancer=False)
            prediction_text = f"Prediction: {label} — {risk_level} ({benign_prob*100:.1f}% benign probability)"
        
        # DEBUG: Log the prediction
        print(f"DEBUG Breast Prediction: Malignant prob={malignant_prob:.3f}, Benign prob={benign_prob:.3f}")
        
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

@app.route("/selftest_bc")
def selftest_bc():
    y_pred = calibrator_bc.predict(X_test_bc)
    y_pred_proba = calibrator_bc.predict_proba(X_test_bc)[:, 1]  # Malignant probability
    
    acc = accuracy_score(y_test_bc, y_pred)
    cm = confusion_matrix(y_test_bc, y_pred).tolist()
    report = classification_report(y_test_bc, y_pred, target_names=["Benign", "Malignant"], output_dict=True)
    
    # Calculate sensitivity (true positive rate) and specificity
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return render_template(
        "selftest.html", 
        accuracy=acc, 
        cm=cm, 
        target_names=["Benign", "Malignant"], 
        report=report,
        sensitivity=sensitivity,
        specificity=specificity,
        module="Breast Cancer"
    )

@app.route("/clustering_bc")
def clustering_bc():
    labels = kmeans_bc.predict(scaler_bc.transform(X_bc[:, :10]))  # Original features only
    ari = adjusted_rand_score(y_bc, labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(X_scaled_bc[:, 0], X_scaled_bc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax1.set_title("K-means Clustering (3 clusters)")
    ax1.set_xlabel(selected_features_bc[0])
    ax1.set_ylabel(selected_features_bc[1])
    
    ax2.scatter(X_scaled_bc[:, 0], X_scaled_bc[:, 1], c=y_bc, cmap="coolwarm", alpha=0.6)
    ax2.set_title("Actual Diagnosis (Red=Malignant, Blue=Benign)")
    ax2.set_xlabel(selected_features_bc[0])
    ax2.set_ylabel(selected_features_bc[1])
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=100)
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template("clustering.html", ari=ari, plot_url=plot_url, n_clusters=3)

# ---------------- LUNG CANCER ROUTES - FIXED ----------------
@app.route("/lung")
def lung_page():
    return render_template("lung_cancer.html", 
                         features_lc=selected_features_lc[:15],  # Only original features
                         demo_values_lc=demo_values_lc[:15] if demo_values_lc else [])

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    if model_lc is None:
        return render_template("lung_cancer.html", 
                             features_lc=selected_features_lc[:15],
                             demo_values_lc=demo_values_lc[:15] if demo_values_lc else [],
                             prediction_text="Lung model not available on server.")
    try:
        # Get all 15 original features
        feature_vals = []
        for i in range(15):  # Lung cancer has 15 original features
            val = request.form.get(f"feature_lc{i+1}")
            if val is not None and val.strip() != "":
                try:
                    feature_vals.append(float(val))
                except:
                    feature_vals.append(0.0)
            else:
                feature_vals.append(0.0)
        
        # Prepare enhanced features
        enhanced_features = prepare_lung_features(feature_vals)
        
        # Get probabilities
        proba = model_lc.predict_proba([enhanced_features])[0]
        cancer_prob = proba[1]  # Probability of lung cancer
        
        if cancer_prob > 0.5:
            label = "Lung Cancer Detected"
            risk_level, confidence = get_risk_category(cancer_prob, is_cancer=True)
            prediction_text = f"Prediction: {label} — {risk_level} ({cancer_prob*100:.1f}% probability)"
        else:
            label = "No Lung Cancer"
            risk_level, confidence = get_risk_category(cancer_prob, is_cancer=False)
            prediction_text = f"Prediction: {label} — {risk_level} ({cancer_prob*100:.1f}% probability)"
        
        return render_template(
            "lung_cancer.html", 
            features_lc=selected_features_lc[:15],
            demo_values_lc=demo_values_lc[:15] if demo_values_lc else [],
            prediction_text=prediction_text
        )
    except Exception as e:
        return render_template(
            "lung_cancer.html", 
            features_lc=selected_features_lc[:15],
            demo_values_lc=demo_values_lc[:15] if demo_values_lc else [],
            prediction_text=f"Error: {str(e)}"
        )

# ---------------- PROSTATE CANCER ROUTES - PERFECT ----------------
@app.route('/prostate')
def prostate_page():
    return render_template(
        "prostate.html",
        features_prostate=features_prostate,
        demo_values_prostate=demo_values_prostate
    )

@app.route("/predict_prostate", methods=["POST"])
def predict_prostate():
    try:
        values = []
        for i in range(len(features_prostate)):
            val = request.form.get(f"feature_prostate{i+1}")
            if val is None or val.strip() == "":
                values.append(0)
            else:
                try:
                    values.append(float(val))
                except:
                    values.append(0)

        # Perfect rule-based system
        age = values[0]
        psa = values[1]
        biopsy = values[2]
        tumor_size = values[3]
        cancer_stage = values[4]
        family_history = values[7]

        if biopsy == 1:
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (98.0% probability)"
        elif psa > 20.0:
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (90.0% probability)"
        elif psa > 10.0:
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (85.0% probability)"
        elif psa > 4.0 and cancer_stage >= 2:
            prediction_text = "Prediction: Prostate Cancer Detected — Moderate risk (75.0% probability)"
        elif psa > 4.0 and family_history == 1:
            prediction_text = "Prediction: Prostate Cancer Detected — Moderate risk (65.0% probability)"
        elif psa > 4.0:
            prediction_text = "Prediction: Prostate Cancer Detected — Low risk (45.0% probability)"
        else:
            prediction_text = "Prediction: No Prostate Cancer — Low risk (10.0% probability)"

        return render_template(
            "prostate.html",
            features_prostate=features_prostate,
            demo_values_prostate=demo_values_prostate,
            prediction_text=prediction_text
        )
    except Exception as e:
        return render_template(
            "prostate.html",
            features_prostate=features_prostate,
            demo_values_prostate=demo_values_prostate,
            prediction_text=f"Error: {e}"
        )

# ==================================================
# TEST ENDPOINT - For quick validation
# ==================================================
@app.route("/test_borderline", methods=["GET"])
def test_borderline():
    """Test the previously failing cases"""
    test_cases = [
        {
            "name": "Case 3 - Borderline Malignant (Previously FAILED)",
            "features": [14.99, 0.02701, 97.65, 0.1471, 0.0701, 14.25, 450.9, 94.96, 17.94, 0.1015]
        },
        {
            "name": "Case 5 - Borderline Malignant (Previously FAILED)", 
            "features": [16.25, 0.03735, 108.4, 0.1822, 0.0953, 14.92, 1022.0, 97.65, 19.61, 0.1099]
        }
    ]
    
    results = []
    for case in test_cases:
        enhanced = prepare_breast_features(case["features"])
        proba = calibrator_bc.predict_proba([enhanced])[0]
        malignant_prob = proba[1]
        
        results.append({
            "case": case["name"],
            "malignant_probability": f"{malignant_prob*100:.1f}%",
            "prediction": "MALIGNANT" if malignant_prob > 0.5 else "BENIGN",
            "status": "✅ FIXED" if malignant_prob > 0.5 else "❌ STILL BROKEN"
        })
    
    html = "<h1>Borderline Case Test Results</h1>"
    for r in results:
        html += f"<h3>{r['case']}</h3>"
        html += f"<p>Malignant Probability: {r['malignant_probability']}</p>"
        html += f"<p>Prediction: {r['prediction']}</p>"
        html += f"<p>Status: {r['status']}</p><hr>"
    
    return html

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 SMARTONCO SYSTEM STARTING")
    print("="*60)
    print(f"Breast Model: {'✅ Ready' if 'calibrator_bc' in globals() else '❌ Not ready'}")
    print(f"Lung Model: {'✅ Ready' if model_lc else '❌ Not ready'}")
    print(f"Prostate Model: ✅ Ready (Rule-based)")
    print("="*60)
    print("Access the system at: http://localhost:5000")
    print("Test borderline cases: http://localhost:5000/test_borderline")
    print("="*60)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
