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
# BREAST CANCER SECTION (unchanged)
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.3, random_state=42)
model_bc = DecisionTreeClassifier(random_state=42)
model_bc.fit(X_train_bc, y_train_bc)
joblib.dump(model_bc, "decision_tree_model_bc.pkl")

demo_values_bc = X_test_bc[0].tolist()
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)
kmeans_bc = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_bc.fit(X_scaled_bc)

# ==================================================
# LUNG CANCER SECTION (unchanged)
# ==================================================
# Make sure file exists in repo root: "survey lung cancer.csv"
try:
    data_lc = pd.read_csv("survey lung cancer.csv")
    # map gender and labels - tolerate missing columns gracefully
    if "GENDER" in data_lc.columns:
        data_lc["GENDER"] = data_lc["GENDER"].map({"M": 1, "F": 0})
    if "LUNG_CANCER" in data_lc.columns:
        data_lc["LUNG_CANCER"] = data_lc["LUNG_CANCER"].map({"NO": 0, "YES": 1})

    if "LUNG_CANCER" in data_lc.columns:
        X_lc = data_lc.drop("LUNG_CANCER", axis=1)
        y_lc = data_lc["LUNG_CANCER"]
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(X_lc, y_lc, test_size=0.3, random_state=42)
    else:
        X_lc = data_lc.copy()
        y_lc = None
        X_train_lc = X_test_lc = y_train_lc = y_test_lc = None
except Exception:
    data_lc = None
    X_lc = None
    y_lc = None
    X_train_lc = X_test_lc = y_train_lc = y_test_lc = None

# Load or train a simple lung model if data is present; else create a fallback
if X_train_lc is not None and y_train_lc is not None:
    model_lc = DecisionTreeClassifier(random_state=42)
    model_lc.fit(X_train_lc, y_train_lc)
    joblib.dump(model_lc, "lung_cancer_model.pkl")
    demo_values_lc = X_test_lc.iloc[0].tolist()
    selected_features_lc = list(X_lc.columns)
    scaler_lc = StandardScaler()
    X_scaled_lc = scaler_lc.fit_transform(X_lc)
    kmeans_lc = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_lc.fit(X_scaled_lc)
else:
    # placeholders to avoid template errors
    model_lc = None
    demo_values_lc = []
    selected_features_lc = []
    scaler_lc = None
    kmeans_lc = None

# ==================================================
# PROSTATE CANCER SECTION (Updated 15 features)
# ==================================================
features_prostate = [
    "Age",
    "Family_History",
    "Race_African_Ancestry",
    "PSA_Level",
    "DRE_Result",
    "Difficulty_Urinating",
    "Weak_Urine_Flow",
    "Blood_in_Urine",
    "Pelvic_Pain",
    "Erectile_Dysfunction",
    "BMI",
    "Smoking_History",
    "Hypertension",
    "Diabetes",
    "Genetic_Risk_Factors"
]

# Placeholder demo values (match features_prostate order)
demo_values_prostate = [60, 1, 1, 8.5, 1, 0, 1, 0, 0, 1, 27.5, 0, 1, 0, 1]

# Try to load dataset & pretrained model if available
data_pc = None
X_pc = None
y_pc = None
X_train_pc = X_test_pc = y_train_pc = y_test_pc = None
scaler_pc = None
kmeans_pc = None
model_prostate = None

# attempt to load training data if present (optional)
if os.path.exists("prostate_cancer.csv"):
    try:
        data_pc = pd.read_csv("prostate_cancer.csv")
        # perform light preprocessing: encode categorical
        for col in data_pc.select_dtypes(include=['object']).columns:
            data_pc[col] = data_pc[col].astype('category').cat.codes

        # if Early_Detection exists, use it as target
        if "Early_Detection" in data_pc.columns:
            X_pc = data_pc.drop(columns=["Early_Detection"], errors='ignore')
            y_pc = data_pc["Early_Detection"]
            X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(X_pc, y_pc, test_size=0.3, random_state=42)
            scaler_pc = StandardScaler()
            X_scaled_pc = scaler_pc.fit_transform(X_pc)
            kmeans_pc = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans_pc.fit(X_scaled_pc)
    except Exception:
        data_pc = None

# attempt to load an existing pre-trained prostate model if file exists
if os.path.exists("prostate_cancer_model.pkl"):
    try:
        model_prostate = joblib.load("prostate_cancer_model.pkl")
    except Exception:
        model_prostate = None

# If model_prostate was not loaded but dataset available, train a quick model as fallback (optional)
if model_prostate is None and X_train_pc is not None and y_train_pc is not None:
    try:
        model_prostate = DecisionTreeClassifier(random_state=42)
        model_prostate.fit(X_train_pc, y_train_pc)
        joblib.dump(model_prostate, "prostate_cancer_model.pkl")
    except Exception:
        model_prostate = None

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc)

# ---------------- BREAST ----------------
@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    feature_vals = [float(request.form.get(f"feature_bc{i+1}")) for i in range(len(selected_features_bc))]
    df = pd.DataFrame([feature_vals], columns=selected_features_bc)
    pred = model_bc.predict(df)[0]
    proba = model_bc.predict_proba(df)[0][0]
    label = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-cancerous)"

    if proba >= 0.80:
        risk = "High risk"
    elif proba >= 0.50:
        risk = "Moderate risk"
    else:
        risk = "Low risk"

    prediction_text = f"Prediction: {label} — {risk} ({proba*100:.1f}% malignant probability)"
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc, prediction_text=prediction_text)

@app.route("/selftest_bc")
def selftest_bc():
    y_pred = model_bc.predict(X_test_bc)
    acc = accuracy_score(y_test_bc, y_pred)
    cm = confusion_matrix(y_test_bc, y_pred).tolist()
    report = classification_report(y_test_bc, y_pred, target_names=data_bc.target_names, output_dict=True)
    return render_template("selftest.html", accuracy=acc, cm=cm, target_names=data_bc.target_names, report=report)

@app.route("/clustering_bc")
def clustering_bc():
    labels = kmeans_bc.predict(scaler_bc.transform(X_bc))
    ari = adjusted_rand_score(y_bc, labels)
    fig, ax = plt.subplots()
    ax.scatter(X_scaled_bc[:, 0], X_scaled_bc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.set_title("Breast Cancer - KMeans Clustering (first 2 features)")
    plt.xlabel(selected_features_bc[0])
    plt.ylabel(selected_features_bc[1])
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template("clustering.html", ari=ari, plot_url=plot_url)

# ---------------- LUNG ----------------
@app.route("/lung")
def lung_page():
    return render_template("lung_cancer.html", features_lc=selected_features_lc, demo_values_lc=demo_values_lc)

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    # defensive: ensure model exists
    if model_lc is None:
        return render_template("lung_cancer.html", features_lc=selected_features_lc, demo_values_lc=demo_values_lc,
                               prediction_text="Lung model not available on server.")
    try:
        feature_vals = [float(request.form.get(f"feature_lc{i+1}")) for i in range(len(selected_features_lc))]
        df = pd.DataFrame([feature_vals], columns=selected_features_lc)
        pred = model_lc.predict(df)[0]
        proba = model_lc.predict_proba(df)[0][1]
        label = "Lung Cancer Detected" if pred == 1 else "No Lung Cancer"

        if proba >= 0.80:
            risk = "High risk"
        elif proba >= 0.50:
            risk = "Moderate risk"
        else:
            risk = "Low risk"

        prediction_text = f"Prediction: {label} — {risk} ({proba*100:.1f}% probability)"
    except Exception as e:
        prediction_text = f"Error during lung prediction: {e}"

    return render_template("lung_cancer.html", features_lc=selected_features_lc, demo_values_lc=demo_values_lc, prediction_text=prediction_text)

@app.route("/selftest_lc")
def selftest_lc():
    if X_test_lc is None:
        return render_template("selftest.html", accuracy=None, cm=None, target_names=None, report=None,
                               message="Lung dataset not available for self-test.")
    y_pred = model_lc.predict(X_test_lc)
    acc = accuracy_score(y_test_lc, y_pred)
    cm = confusion_matrix(y_test_lc, y_pred).tolist()
    report = classification_report(y_test_lc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    return render_template("selftest.html", accuracy=acc, cm=cm, target_names=["No Cancer", "Cancer"], report=report)

@app.route("/clustering_lc")
def clustering_lc():
    if X_lc is None or kmeans_lc is None:
        return render_template("clustering.html", ari=None, plot_url=None, message="Lung data not available for clustering.")
    labels = kmeans_lc.predict(scaler_lc.transform(X_lc))
    ari = adjusted_rand_score(y_lc, labels) if y_lc is not None else None
    fig, ax = plt.subplots()
    ax.scatter(X_scaled_lc[:, 0], X_scaled_lc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.set_title("Lung Cancer - KMeans Clustering (first 2 features)")
    plt.xlabel(selected_features_lc[0] if len(selected_features_lc) > 0 else "f1")
    plt.ylabel(selected_features_lc[1] if len(selected_features_lc) > 1 else "f2")
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template("clustering.html", ari=ari, plot_url=plot_url)

# ---------------- PROSTATE ----------------
@app.route('/prostate')
def prostate_page():
    return render_template(
        "prostate.html",
        features_pc=features_prostate,
        demo_values_pc=demo_values_prostate
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

        # Convert to DataFrame using the same columns used for training
        df = pd.DataFrame([values], columns=features_prostate)

        # ✅ Fix: Align columns with model’s expected features if model has .feature_names_in_
        if hasattr(model_prostate, "feature_names_in_"):
            expected_cols = list(model_prostate.feature_names_in_)
            # Add any missing columns (fill with 0)
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            # Keep only the model's expected columns (correct order)
            df = df[expected_cols]

        # Predict safely
        pred = model_prostate.predict(df)[0]
        try:
            proba = model_prostate.predict_proba(df)[0][1]
        except Exception:
            proba = None

        label = "Prostate Cancer Detected" if pred == 1 else "No Prostate Cancer"

        if proba is not None:
            if proba >= 0.80:
                risk = "High risk"
            elif proba >= 0.50:
                risk = "Moderate risk"
            else:
                risk = "Low risk"
            prob_text = f" ({proba*100:.1f}% probability)"
        else:
            risk = ""
            prob_text = ""

        prediction_text = f"Prediction: {label} — {risk}{prob_text}"

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
        )@app.route("/predict_prostate", methods=["POST"])
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

        # Convert to DataFrame using the same columns used for training
        df = pd.DataFrame([values], columns=features_prostate)

        # ✅ Fix: Align columns with model’s expected features if model has .feature_names_in_
        if hasattr(model_prostate, "feature_names_in_"):
            expected_cols = list(model_prostate.feature_names_in_)
            # Add any missing columns (fill with 0)
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            # Keep only the model's expected columns (correct order)
            df = df[expected_cols]

        # Predict safely
        pred = model_prostate.predict(df)[0]
        try:
            proba = model_prostate.predict_proba(df)[0][1]
        except Exception:
            proba = None

        label = "Prostate Cancer Detected" if pred == 1 else "No Prostate Cancer"

        if proba is not None:
            if proba >= 0.80:
                risk = "High risk"
            elif proba >= 0.50:
                risk = "Moderate risk"
            else:
                risk = "Low risk"
            prob_text = f" ({proba*100:.1f}% probability)"
        else:
            risk = ""
            prob_text = ""

        prediction_text = f"Prediction: {label} — {risk}{prob_text}"

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


@app.route("/selftest_pc")
def selftest_pc():
    if X_test_pc is None or y_test_pc is None:
        return render_template("selftest.html", accuracy=None, cm=None, target_names=None, report=None,
                               message="Prostate dataset not available for self-test.")
    y_pred = model_prostate.predict(X_test_pc)
    acc = accuracy_score(y_test_pc, y_pred)
    cm = confusion_matrix(y_test_pc, y_pred).tolist()
    report = classification_report(y_test_pc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    return render_template("selftest.html", accuracy=acc, cm=cm, target_names=["No Cancer", "Cancer"], report=report)

@app.route("/clustering_pc")
def clustering_pc():
    if X_pc is None or kmeans_pc is None:
        return render_template("clustering.html", ari=None, plot_url=None, message="Prostate data not available for clustering.")
    labels = kmeans_pc.predict(scaler_pc.transform(X_pc))
    ari = adjusted_rand_score(y_pc, labels) if y_pc is not None else None
    fig, ax = plt.subplots()
    ax.scatter(X_scaled_pc[:, 0], X_scaled_pc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.set_title("Prostate Cancer - KMeans Clustering (first 2 features)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template("clustering.html", ari=ari, plot_url=plot_url)

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
