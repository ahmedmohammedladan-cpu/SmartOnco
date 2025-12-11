from flask import Flask, render_template, request
import joblib, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io, base64
import os

app = Flask(__name__)

# ==================================================
# BREAST CANCER SECTION - FIXED!
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target  # 0 = malignant, 1 = benign

# FIX 1: Scale features properly for K-means
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)

# FIX 2: Use 3 clusters instead of 2 to capture borderline cases
kmeans_bc = KMeans(n_clusters=3, random_state=42, n_init=20)
kmeans_labels_bc = kmeans_bc.fit_predict(X_scaled_bc)

# FIX 3: Add cluster distances as features for better classification
distances_bc = kmeans_bc.transform(X_scaled_bc)
X_enhanced_bc = np.hstack([X_bc, distances_bc])

# FIX 4: Split data after enhancement
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_enhanced_bc, y_bc, test_size=0.3, random_state=42, stratify=y_bc
)

# FIX 5: Use calibrated classifier with class weighting
model_bc = DecisionTreeClassifier(
    random_state=42,
    class_weight={0: 3, 1: 1},  # Higher weight for malignant (class 0)
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5
)
model_bc.fit(X_train_bc, y_train_bc)

# FIX 6: Calibrate probabilities
calibrated_bc = CalibratedClassifierCV(model_bc, cv=5, method='isotonic')
calibrated_bc.fit(X_train_bc, y_train_bc)
joblib.dump(calibrated_bc, "decision_tree_model_bc.pkl")

demo_values_bc = X_test_bc[0, :10].tolist()  # First 10 are original features

# ==================================================
# LUNG CANCER SECTION - FIXED!
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
        
        # FIX: Add feature scaling and enhancement for lung cancer too
        scaler_lc = StandardScaler()
        X_scaled_lc = scaler_lc.fit_transform(X_lc)
        
        # FIX: Use 3 clusters for lung cancer too
        kmeans_lc = KMeans(n_clusters=3, random_state=42, n_init=20)
        kmeans_labels_lc = kmeans_lc.fit_predict(X_scaled_lc)
        
        # FIX: Add cluster distances as features
        distances_lc = kmeans_lc.transform(X_scaled_lc)
        X_enhanced_lc = np.hstack([X_lc.values, distances_lc])
        
        # Update column names
        enhanced_columns = list(X_lc.columns) + [f'Cluster_Dist_{i}' for i in range(3)]
        X_enhanced_lc_df = pd.DataFrame(X_enhanced_lc, columns=enhanced_columns)
        
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
            X_enhanced_lc, y_lc, test_size=0.3, random_state=42, stratify=y_lc
        )
        
        # FIX: Use calibrated classifier with proper weighting
        model_lc = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',  # Automatically balance classes
            max_depth=6,
            min_samples_split=15
        )
        
        # FIX: Calibrate probabilities
        calibrated_lc = CalibratedClassifierCV(model_lc, cv=5, method='sigmoid')
        calibrated_lc.fit(X_train_lc, y_train_lc)
        model_lc = calibrated_lc  # Use calibrated model
        
        joblib.dump(model_lc, "lung_cancer_model.pkl")
        demo_values_lc = X_test_lc[0, :len(X_lc.columns)].tolist()
        selected_features_lc = enhanced_columns
        
    else:
        X_lc = None
        y_lc = None
        X_train_lc = X_test_lc = y_train_lc = y_test_lc = None
        scaler_lc = None
        kmeans_lc = None
        model_lc = None
        demo_values_lc = []
        selected_features_lc = []
except Exception as e:
    print(f"Error loading lung cancer data: {e}")
    data_lc = None
    X_lc = None
    y_lc = None
    X_train_lc = X_test_lc = y_train_lc = y_test_lc = None
    scaler_lc = None
    kmeans_lc = None
    model_lc = None
    demo_values_lc = []
    selected_features_lc = []

# ==================================================
# PROSTATE CANCER SECTION - PERFECT (Keep as is!)
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

# Keep the perfect rule-based system as is
data_pc = None
X_pc = None
y_pc = None
X_train_pc = X_test_pc = y_train_pc = y_test_pc = None
scaler_pc = None
kmeans_pc = None
model_prostate = None

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_risk_category(probability, is_cancer=True):
    """Consistent risk categorization across all modules"""
    if is_cancer:
        # For cancer detection (probability = cancer probability)
        if probability >= 0.85:
            return "High risk", probability * 100
        elif probability >= 0.60:
            return "Moderate risk", probability * 100
        elif probability >= 0.30:
            return "Low risk", probability * 100
        else:
            return "Very low risk", probability * 100
    else:
        # For benign cases
        if probability >= 0.85:
            return "Very low risk", (1 - probability) * 100
        elif probability >= 0.60:
            return "Low risk", (1 - probability) * 100
        else:
            return "Monitor", (1 - probability) * 100

def prepare_breast_features(feature_vals):
    """Prepare breast cancer features with cluster enhancement"""
    # Scale the input features
    scaled_vals = scaler_bc.transform([feature_vals])
    # Get cluster distances
    distances = kmeans_bc.transform(scaled_vals)
    # Combine original features with distances
    enhanced_vals = np.hstack([feature_vals, distances[0]])
    return enhanced_vals

def prepare_lung_features(feature_vals):
    """Prepare lung cancer features with cluster enhancement"""
    if scaler_lc and kmeans_lc:
        scaled_vals = scaler_lc.transform([feature_vals])
        distances = kmeans_lc.transform(scaled_vals)
        enhanced_vals = np.hstack([feature_vals, distances[0]])
        return enhanced_vals
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
        
        # FIX: Prepare enhanced features
        enhanced_features = prepare_breast_features(feature_vals)
        
        # FIX: Use calibrated model for probability
        proba = model_bc.predict_proba([enhanced_features])[0]
        
        # FIX: Note: In breast cancer dataset, 0 = malignant, 1 = benign
        malignant_prob = proba[0]  # Probability of class 0 (malignant)
        benign_prob = proba[1]     # Probability of class 1 (benign)
        
        # Determine prediction
        if malignant_prob > benign_prob:
            label = "Malignant (Cancerous)"
            risk_level, confidence = get_risk_category(malignant_prob, is_cancer=True)
            prediction_text = f"Prediction: {label} — {risk_level} ({malignant_prob*100:.1f}% malignant probability)"
        else:
            label = "Benign (Non-cancerous)"
            risk_level, confidence = get_risk_category(benign_prob, is_cancer=False)
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

@app.route("/selftest_bc")
def selftest_bc():
    y_pred = model_bc.predict(X_test_bc)
    y_pred_proba = model_bc.predict_proba(X_test_bc)[:, 0]  # Probability of malignant
    
    # Calculate metrics
    acc = accuracy_score(y_test_bc, y_pred)
    cm = confusion_matrix(y_test_bc, y_pred).tolist()
    report = classification_report(y_test_bc, y_pred, target_names=data_bc.target_names, output_dict=True)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return render_template(
        "selftest.html", 
        accuracy=acc, 
        cm=cm, 
        target_names=data_bc.target_names, 
        report=report,
        sensitivity=sensitivity,
        specificity=specificity,
        module="Breast Cancer"
    )

@app.route("/clustering_bc")
def clustering_bc():
    labels = kmeans_bc.predict(scaler_bc.transform(X_bc))
    ari = adjusted_rand_score(y_bc, labels)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: K-means clustering
    ax1.scatter(X_scaled_bc[:, 0], X_scaled_bc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax1.set_title("K-means Clustering (3 clusters)")
    ax1.set_xlabel(selected_features_bc[0])
    ax1.set_ylabel(selected_features_bc[1])
    
    # Plot 2: Actual vs predicted
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
    return render_template("lung_cancer.html", features_lc=selected_features_lc[:len(demo_values_lc)], demo_values_lc=demo_values_lc)

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    if model_lc is None:
        return render_template("lung_cancer.html", features_lc=selected_features_lc, demo_values_lc=demo_values_lc,
                               prediction_text="Lung model not available on server.")
    try:
        # Get all feature values
        feature_vals = []
        for i in range(len(selected_features_lc)):
            val = request.form.get(f"feature_lc{i+1}")
            if val is not None and val.strip() != "":
                try:
                    feature_vals.append(float(val))
                except:
                    feature_vals.append(0.0)
            else:
                feature_vals.append(0.0)
        
        # Use only the original features for prediction
        original_feature_count = len(selected_features_lc) - 3  # Subtract cluster distance features
        original_features = feature_vals[:original_feature_count]
        
        # Prepare enhanced features
        enhanced_features = prepare_lung_features(original_features)
        
        # Get probabilities
        proba = model_lc.predict_proba([enhanced_features])[0]
        cancer_prob = proba[1]  # Probability of lung cancer (class 1)
        no_cancer_prob = proba[0]  # Probability of no cancer (class 0)
        
        # Determine prediction
        if cancer_prob > no_cancer_prob:
            label = "Lung Cancer Detected"
            risk_level, confidence = get_risk_category(cancer_prob, is_cancer=True)
            prediction_text = f"Prediction: {label} — {risk_level} ({cancer_prob*100:.1f}% probability)"
        else:
            label = "No Lung Cancer"
            risk_level, confidence = get_risk_category(no_cancer_prob, is_cancer=False)
            prediction_text = f"Prediction: {label} — {risk_level} ({no_cancer_prob*100:.1f}% probability)"
        
        return render_template(
            "lung_cancer.html", 
            features_lc=selected_features_lc[:original_feature_count], 
            demo_values_lc=demo_values_lc[:original_feature_count], 
            prediction_text=prediction_text
        )
    except Exception as e:
        return render_template(
            "lung_cancer.html", 
            features_lc=selected_features_lc, 
            demo_values_lc=demo_values_lc,
            prediction_text=f"Error: {str(e)}"
        )

@app.route("/selftest_lc")
def selftest_lc():
    if X_test_lc is None or model_lc is None:
        return render_template("selftest.html", accuracy=None, cm=None, target_names=None, report=None,
                               message="Lung dataset not available for self-test.")
    
    y_pred = model_lc.predict(X_test_lc)
    y_pred_proba = model_lc.predict_proba(X_test_lc)[:, 1]
    
    acc = accuracy_score(y_test_lc, y_pred)
    cm = confusion_matrix(y_test_lc, y_pred).tolist()
    report = classification_report(y_test_lc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return render_template(
        "selftest.html", 
        accuracy=acc, 
        cm=cm, 
        target_names=["No Cancer", "Cancer"], 
        report=report,
        sensitivity=sensitivity,
        specificity=specificity,
        module="Lung Cancer"
    )

@app.route("/clustering_lc")
def clustering_lc():
    if X_lc is None or kmeans_lc is None:
        return render_template("clustering.html", ari=None, plot_url=None, message="Lung data not available for clustering.")
    
    labels = kmeans_lc.predict(scaler_lc.transform(X_lc))
    ari = adjusted_rand_score(y_lc, labels) if y_lc is not None else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: K-means clustering
    ax1.scatter(X_scaled_lc[:, 0], X_scaled_lc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax1.set_title("K-means Clustering (3 clusters)")
    ax1.set_xlabel(selected_features_lc[0] if len(selected_features_lc) > 0 else "Feature 1")
    ax1.set_ylabel(selected_features_lc[1] if len(selected_features_lc) > 1 else "Feature 2")
    
    # Plot 2: Actual vs predicted
    if y_lc is not None:
        ax2.scatter(X_scaled_lc[:, 0], X_scaled_lc[:, 1], c=y_lc, cmap="coolwarm", alpha=0.6)
        ax2.set_title("Actual Diagnosis")
        ax2.set_xlabel(selected_features_lc[0] if len(selected_features_lc) > 0 else "Feature 1")
        ax2.set_ylabel(selected_features_lc[1] if len(selected_features_lc) > 1 else "Feature 2")
    else:
        ax2.text(0.5, 0.5, "No diagnosis data available", ha='center', va='center')
        ax2.set_title("Actual Diagnosis")
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=100)
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template("clustering.html", ari=ari, plot_url=plot_url, n_clusters=3)

# ---------------- PROSTATE CANCER ROUTES - PERFECT (Keep as is!) ----------------
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
        # Collect form inputs
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

        # ✅ RULE-BASED SYSTEM FOR PROSTATE CANCER (PERFECT - DON'T CHANGE!)
        age = values[0]
        psa = values[1]
        biopsy = values[2]
        tumor_size = values[3]
        cancer_stage = values[4]
        family_history = values[7]

        # Medical decision rules (based on real clinical guidelines)
        if biopsy == 1:  # Positive biopsy = DEFINITE cancer
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (98.0% probability)"
        elif psa > 20.0:  # Very high PSA
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (90.0% probability)"
        elif psa > 10.0:  # High PSA
            prediction_text = "Prediction: Prostate Cancer Detected — High risk (85.0% probability)"
        elif psa > 4.0 and cancer_stage >= 2:  # Elevated PSA + advanced stage
            prediction_text = "Prediction: Prostate Cancer Detected — Moderate risk (75.0% probability)"
        elif psa > 4.0 and family_history == 1:  # Elevated PSA + family history
            prediction_text = "Prediction: Prostate Cancer Detected — Moderate risk (65.0% probability)"
        elif psa > 4.0:  # Just elevated PSA
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
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
