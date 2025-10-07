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

app = Flask(__name__)

# ==================================================
# BREAST CANCER SECTION
# ==================================================
selected_features_bc = [
    "worst radius",
    "mean concave points",
    "worst perimeter",
    "mean concavity",
    "worst concave points",
    "mean radius",
    "worst area",
    "mean perimeter",
    "mean texture",
    "worst smoothness"
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
# LUNG CANCER SECTION
# ==================================================
data_lc = pd.read_csv("survey lung cancer.csv")
data_lc["GENDER"] = data_lc["GENDER"].map({"M": 1, "F": 0})
data_lc["LUNG_CANCER"] = data_lc["LUNG_CANCER"].map({"NO": 0, "YES": 1})

X_lc = data_lc.drop("LUNG_CANCER", axis=1)
y_lc = data_lc["LUNG_CANCER"]

X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(X_lc, y_lc, test_size=0.3, random_state=42)
model_lc = DecisionTreeClassifier(random_state=42)
model_lc.fit(X_train_lc, y_train_lc)
joblib.dump(model_lc, "lung_cancer_model.pkl")

demo_values_lc = X_test_lc.iloc[0].tolist()
selected_features_lc = list(X_lc.columns)

scaler_lc = StandardScaler()
X_scaled_lc = scaler_lc.fit_transform(X_lc)
kmeans_lc = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_lc.fit(X_scaled_lc)

# ==================================================
# PROSTATE CANCER SECTION
# ==================================================
data_pc = pd.read_csv("prostate_cancer.csv")

# Encode categorical columns
for col in data_pc.select_dtypes(include=['object']).columns:
    data_pc[col] = data_pc[col].astype('category').cat.codes

X_pc = data_pc.drop(columns=["Early_Detection"], errors='ignore')
y_pc = data_pc["Early_Detection"]

X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(X_pc, y_pc, test_size=0.3, random_state=42)

# Load pre-trained model
model_pc = joblib.load("prostate_cancer_model.pkl")

demo_values_pc = X_test_pc.iloc[0].tolist()
selected_features_pc = list(X_pc.columns)

scaler_pc = StandardScaler()
X_scaled_pc = scaler_pc.fit_transform(X_pc)
kmeans_pc = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_pc.fit(X_scaled_pc)

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template("index.html", features_bc=selected_features_bc, demo_values_bc=demo_values_bc)

# ---------------- BREAST ----------------
@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    n = len(selected_features_bc)
    feature_vals = [float(request.form.get(f"feature_bc{i+1}")) for i in range(n)]
    df = pd.DataFrame([feature_vals], columns=selected_features_bc)
    pred = model_bc.predict(df)[0]
    proba = model_bc.predict_proba(df)[0][0]
    label = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-cancerous)"

    if proba >= 0.80: risk = "High risk"
    elif proba >= 0.50: risk = "Moderate risk"
    else: risk = "Low risk"

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
    n = len(selected_features_lc)
    feature_vals = [float(request.form.get(f"feature_lc{i+1}")) for i in range(n)]
    df = pd.DataFrame([feature_vals], columns=selected_features_lc)
    pred = model_lc.predict(df)[0]
    proba = model_lc.predict_proba(df)[0][1]
    label = "Lung Cancer Detected" if pred == 1 else "No Lung Cancer"

    if proba >= 0.80: risk = "High risk"
    elif proba >= 0.50: risk = "Moderate risk"
    else: risk = "Low risk"

    prediction_text = f"Prediction: {label} — {risk} ({proba*100:.1f}% probability)"
    return render_template("lung_cancer.html", features_lc=selected_features_lc, demo_values_lc=demo_values_lc, prediction_text=prediction_text)

@app.route("/selftest_lc")
def selftest_lc():
    y_pred = model_lc.predict(X_test_lc)
    acc = accuracy_score(y_test_lc, y_pred)
    cm = confusion_matrix(y_test_lc, y_pred).tolist()
    report = classification_report(y_test_lc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    return render_template("selftest.html", accuracy=acc, cm=cm, target_names=["No Cancer", "Cancer"], report=report)

@app.route("/clustering_lc")
def clustering_lc():
    labels = kmeans_lc.predict(scaler_lc.transform(X_lc))
    ari = adjusted_rand_score(y_lc, labels)
    fig, ax = plt.subplots()
    ax.scatter(X_scaled_lc[:, 0], X_scaled_lc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.set_title("Lung Cancer - KMeans Clustering (first 2 features)")
    plt.xlabel(selected_features_lc[0])
    plt.ylabel(selected_features_lc[1])
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template("clustering.html", ari=ari, plot_url=plot_url)

# ---------------- PROSTATE ----------------
@app.route("/prostate")
def prostate_page():
    return render_template("prostate.html", features_pc=selected_features_pc, demo_values_pc=demo_values_pc)

@app.route("/predict_pc", methods=["POST"])
def predict_pc():
    n = len(selected_features_pc)
    feature_vals = [float(request.form.get(f"feature_pc{i+1}")) for i in range(n)]
    df = pd.DataFrame([feature_vals], columns=selected_features_pc)

    pred = model_pc.predict(df)[0]
    proba = model_pc.predict_proba(df)[0][1]
    label = "Prostate Cancer Detected" if pred == 1 else "No Prostate Cancer"

    if proba >= 0.80: risk = "High risk"
    elif proba >= 0.50: risk = "Moderate risk"
    else: risk = "Low risk"

    prediction_text = f"Prediction: {label} — {risk} ({proba*100:.1f}% probability)"
    return render_template("prostate.html", features_pc=selected_features_pc, demo_values_pc=demo_values_pc, prediction_text=prediction_text)

@app.route("/selftest_pc")
def selftest_pc():
    y_pred = model_pc.predict(X_test_pc)
    acc = accuracy_score(y_test_pc, y_pred)
    cm = confusion_matrix(y_test_pc, y_pred).tolist()
    report = classification_report(
        y_test_pc,
        y_pred,
        target_names=["No Cancer", "Cancer"],
        output_dict=True
    )

    return render_template(
        "selftest.html",
        accuracy=acc,
        cm=cm,
        target_names=["No Cancer", "Cancer"],
        report=report
    )

@app.route("/clustering_pc")
def clustering_pc():
    labels = kmeans_pc.predict(scaler_pc.transform(X_pc))
    ari = adjusted_rand_score(y_pc, labels)
    fig, ax = plt.subplots()
    ax.scatter(X_scaled_pc[:, 0], X_scaled_pc[:, 1], c=labels, cmap="viridis", alpha=0.6)
    ax.set_title("Prostate Cancer - KMeans Clustering (first 2 features)")
    plt.xlabel(selected_features_pc[0])
    plt.ylabel(selected_features_pc[1])
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template("clustering.html", ari=ari, plot_url=plot_url)

# ==================================================
if __name__ == "__main__":
    app.run(debug=True)
