from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# ==================================================
# BREAST CANCER MODEL
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

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)

model_bc = DecisionTreeClassifier(random_state=42)
model_bc.fit(X_train_bc, y_train_bc)

demo_values_bc = X_test_bc[0].tolist()

# Initialize clustering for breast cancer
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)
kmeans_bc = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_bc.fit(X_scaled_bc)

# ==================================================
# SIMULATED DATA FOR LUNG AND PROSTATE (for clustering/self-test)
# ==================================================
# Create simple simulated data for lung cancer (15 features as per your form)
np.random.seed(42)
n_samples = 300

# Lung cancer simulation
X_lung_sim = np.random.randn(n_samples, 15)
# Add some structure
X_lung_sim[:, 1] += np.random.randn(n_samples) * 0.5  # Age-like feature
X_lung_sim[:, 2] += np.random.randn(n_samples) * 0.5  # Smoking-like feature
X_lung_sim[:, 11] += np.random.randn(n_samples) * 0.5  # Coughing-like feature
X_lung_sim[:, 14] += np.random.randn(n_samples) * 0.5  # Chest pain-like feature

y_lung_sim = np.zeros(n_samples, dtype=int)
score = X_lung_sim[:, 1] + X_lung_sim[:, 2] * 1.5 + X_lung_sim[:, 11] + X_lung_sim[:, 14]
y_lung_sim[score > 1] = 1
y_lung_sim[score > 2.5] = 2

# Prostate cancer simulation
X_prostate_sim = np.random.randn(n_samples, 10)
X_prostate_sim[:, 1] += np.random.randn(n_samples) * 0.5  # PSA-like feature
X_prostate_sim[:, 2] += np.random.randn(n_samples) * 0.5  # Biopsy-like feature
y_prostate_sim = ((X_prostate_sim[:, 1] > 0.5) | (X_prostate_sim[:, 2] > 0.7)).astype(int)

# Initialize clustering for lung and prostate
scaler_lung = StandardScaler()
X_scaled_lung = scaler_lung.fit_transform(X_lung_sim)
kmeans_lung = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_lung.fit(X_scaled_lung)

scaler_prostate = StandardScaler()
X_scaled_prostate = scaler_prostate.fit_transform(X_prostate_sim)
kmeans_prostate = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_prostate.fit(X_scaled_prostate)

# ==================================================
# MANUAL RULES – BREAST CANCER
# ==================================================
def predict_breast_manual(features):
    worst_radius = features[0]
    worst_area = features[6]
    worst_concave_pts = features[4]

    if worst_radius > 18 or worst_area > 1000:
        return "Malignant (Cancerous)", "High"

    if worst_radius > 14 and worst_concave_pts > 0.05:
        return "Malignant (Cancerous)", "Moderate"

    if worst_radius < 12 and worst_area < 500:
        return "Benign (Non-Cancerous)", "Low"

    return "ml_model", None

# ==================================================
# LUNG CANCER – MANUAL RULE SYSTEM
# ==================================================
def predict_lung_manual(features):
    age = int(features[1])
    smoking = int(features[2])
    coughing = int(features[11])
    chest_pain = int(features[14])

    risk_score = 0
    if age > 60:
        risk_score += 3
    if smoking == 2:
        risk_score += 4
    if coughing == 2:
        risk_score += 2
    if chest_pain == 2:
        risk_score += 3

    if risk_score >= 7:
        return "Cancer Detected", "High"
    elif risk_score >= 4:
        return "Suspicious Findings", "Moderate"
    else:
        return "No Cancer Detected", "Low"

# ==================================================
# RESULT FORMATTING
# ==================================================
def format_result(prediction, risk):
    """
    Returns a professionally formatted result separated by comma and space
    """
    if "Malignant" in prediction or "Cancer Detected" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    elif "Suspicious" in prediction:
        recommendation = "Seek professional medical consultation for confirmation."
    else:
        recommendation = "Routine medical checkups are advised."

    # Join all info in one line
    result = f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"
    return result

# ==================================================
# CLUSTERING FUNCTIONS
# ==================================================
def create_clustering_plot(data_scaled, labels, true_labels, title):
    """Create clustering visualization plot"""
    plt.figure(figsize=(12, 5))
    
    # True labels plot
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], 
                          c=true_labels, cmap='viridis', 
                          alpha=0.6, edgecolors='w', s=50)
    plt.colorbar(scatter1, label='True Labels')
    plt.xlabel('Standardized Feature 1')
    plt.ylabel('Standardized Feature 2')
    plt.title(f'{title} - True Labels')
    
    # Cluster labels plot
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], 
                          c=labels, cmap='viridis', 
                          alpha=0.6, edgecolors='w', s=50)
    plt.colorbar(scatter2, label='Cluster Labels')
    plt.xlabel('Standardized Feature 1')
    plt.ylabel('Standardized Feature 2')
    plt.title(f'{title} - K-means Clustering')
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# ==================================================
# WEB ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc
    )

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    feature_vals = [
        float(request.form.get(f"feature_bc{i+1}"))
        for i in range(len(selected_features_bc))
    ]

    prediction, risk = predict_breast_manual(feature_vals)

    if prediction == "ml_model":
        df = pd.DataFrame([feature_vals], columns=selected_features_bc)
        pred = model_bc.predict(df)[0]
        prediction = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-Cancerous)"
        risk = "Moderate"

    prediction_text = format_result(prediction, risk)

    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=prediction_text
    )

@app.route("/lung")
def lung_page():
    return render_template("lung_cancer.html")

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    feature_vals = [
        float(request.form.get(f"feature_lc{i+1}", 1))
        for i in range(15)
    ]

    prediction, risk = predict_lung_manual(feature_vals)
    prediction_text = format_result(prediction, risk)

    return render_template(
        "lung_cancer.html",
        prediction_text=prediction_text
    )

@app.route("/prostate")
def prostate_page():
    return render_template("prostate.html")

@app.route("/predict_prostate", methods=["POST"])
def predict_prostate():
    psa = float(request.form.get("feature_prostate2", 0))
    biopsy = int(request.form.get("feature_prostate3", 0))

    if biopsy == 1 or psa > 10:
        prediction = "Prostate Cancer Detected"
        risk = "High"
    else:
        prediction = "No Prostate Cancer Detected"
        risk = "Low"

    prediction_text = format_result(prediction, risk)

    return render_template(
        "prostate.html",
        prediction_text=prediction_text
    )

# ==================================================
# NEW ROUTES FOR SELF-TEST
# ==================================================
@app.route("/selftest_bc")
def selftest_bc():
    """Self-test for Breast Cancer Model"""
    y_pred = model_bc.predict(X_test_bc)
    acc = accuracy_score(y_test_bc, y_pred)
    cm = confusion_matrix(y_test_bc, y_pred).tolist()
    report = classification_report(y_test_bc, y_pred, target_names=data_bc.target_names, output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in data_bc.target_names:
        formatted_report[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1-score': report[cls]['f1-score'],
            'support': int(report[cls]['support'])
        }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=data_bc.target_names, 
                         report=formatted_report)

@app.route("/selftest_lc")
def selftest_lc():
    """Self-test for Lung Cancer Model"""
    # For lung cancer, we'll use simulated data and a simple model
    model_lc = DecisionTreeClassifier(random_state=42)
    X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
        X_lung_sim, y_lung_sim, test_size=0.2, random_state=42
    )
    model_lc.fit(X_train_lc, y_train_lc)
    
    y_pred = model_lc.predict(X_test_lc)
    acc = accuracy_score(y_test_lc, y_pred)
    cm = confusion_matrix(y_test_lc, y_pred).tolist()
    report = classification_report(y_test_lc, y_pred, target_names=["No Cancer", "Suspicious", "Cancer"], output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in ["No Cancer", "Suspicious", "Cancer"]:
        if cls in report:
            formatted_report[cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': int(report[cls]['support'])
            }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=["No Cancer", "Suspicious", "Cancer"], 
                         report=formatted_report)

@app.route("/selftest_pc")
def selftest_pc():
    """Self-test for Prostate Cancer"""
    # For prostate cancer, we'll use simulated data and a simple model
    model_pc = DecisionTreeClassifier(random_state=42)
    X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(
        X_prostate_sim, y_prostate_sim, test_size=0.2, random_state=42
    )
    model_pc.fit(X_train_pc, y_train_pc)
    
    y_pred = model_pc.predict(X_test_pc)
    acc = accuracy_score(y_test_pc, y_pred)
    cm = confusion_matrix(y_test_pc, y_pred).tolist()
    report = classification_report(y_test_pc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in ["No Cancer", "Cancer"]:
        formatted_report[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1-score': report[cls]['f1-score'],
            'support': int(report[cls]['support'])
        }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=["No Cancer", "Cancer"], 
                         report=formatted_report)

# ==================================================
# NEW ROUTES FOR CLUSTERING
# ==================================================
@app.route("/clustering_bc")
def clustering_bc():
    """Clustering visualization for Breast Cancer"""
    labels = kmeans_bc.predict(X_scaled_bc)
    ari = adjusted_rand_score(y_bc, labels)
    silhouette = silhouette_score(X_scaled_bc, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_bc, labels, y_bc, "Breast Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Breast",
        n_clusters=2,
        silhouette_score=silhouette
    )

@app.route("/clustering_lc")
def clustering_lc():
    """Clustering visualization for Lung Cancer"""
    labels = kmeans_lung.predict(X_scaled_lung)
    ari = adjusted_rand_score(y_lung_sim, labels)
    silhouette = silhouette_score(X_scaled_lung, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_lung, labels, y_lung_sim, "Lung Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Lung",
        n_clusters=3,
        silhouette_score=silhouette
    )

@app.route("/clustering_pc")
def clustering_pc():
    """Clustering visualization for Prostate Cancer"""
    labels = kmeans_prostate.predict(X_scaled_prostate)
    ari = adjusted_rand_score(y_prostate_sim, labels)
    silhouette = silhouette_score(X_scaled_prostate, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_prostate, labels, y_prostate_sim, "Prostate Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Prostate",
        n_clusters=2,
        silhouette_score=silhouette
    )

# ==================================================
# API ROUTE
# ==================================================
@app.route("/api/predict/breast", methods=["POST"])
def api_predict_breast():
    try:
        data = request.get_json()
        feature_vals = data.get("features", [])

        if len(feature_vals) != len(selected_features_bc):
            return jsonify({"error": "Invalid feature count"}), 400

        prediction, risk = predict_breast_manual(feature_vals)

        if prediction == "ml_model":
            df = pd.DataFrame([feature_vals], columns=selected_features_bc)
            pred = model_bc.predict(df)[0]
            prediction = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-Cancerous)"
            risk = "Moderate"

        full_result = format_result(prediction, risk)

        response = {
            "prediction": prediction,
            "risk_level": risk,
            "recommendation": full_result.split(", ")[2],  # only recommendation
            "full_result": full_result,
            "disclaimer": "This system is for decision support only and does not replace professional medical diagnosis."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models": ["breast", "lung", "prostate"],
        "endpoints": {
            "self_test": "/selftest_bc, /selftest_lc, /selftest_pc",
            "clustering": "/clustering_bc, /clustering_lc, /clustering_pc"
        }
    })

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    print("🚀 SmartOnco System Started")
    print("✅ Breast Cancer: Model loaded")
    print("✅ Lung Cancer: Simulated data ready")
    print("✅ Prostate Cancer: Simulated data ready")
    print("✅ Clustering: Available at /clustering_bc, /clustering_lc, /clustering_pc")
    print("✅ Self-Test: Available at /selftest_bc, /selftest_lc, /selftest_pc")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
