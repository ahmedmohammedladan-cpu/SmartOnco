from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

# =========================
# Gemini AI Import
# =========================
import google.generativeai as genai

# =========================
# Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# Set your Gemini API Key
# =========================
# Make sure to set this as an environment variable in Render:
# GENAI_API_KEY = <your_key>
genai.api_key = os.environ.get("GENAI_API_KEY", "")

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
# SIMULATED DATA FOR LUNG AND PROSTATE
# ==================================================
np.random.seed(42)
n_samples = 300

# Lung cancer simulation
X_lung_sim = np.random.randn(n_samples, 15)
X_lung_sim[:, 1] += np.random.randn(n_samples) * 0.5
X_lung_sim[:, 2] += np.random.randn(n_samples) * 0.5
X_lung_sim[:, 11] += np.random.randn(n_samples) * 0.5
X_lung_sim[:, 14] += np.random.randn(n_samples) * 0.5
y_lung_sim = np.zeros(n_samples, dtype=int)
score = X_lung_sim[:, 1] + X_lung_sim[:, 2] * 1.5 + X_lung_sim[:, 11] + X_lung_sim[:, 14]
y_lung_sim[score > 1] = 1
y_lung_sim[score > 2.5] = 2

# Prostate cancer simulation
X_prostate_sim = np.random.randn(n_samples, 10)
X_prostate_sim[:, 1] += np.random.randn(n_samples) * 0.5
X_prostate_sim[:, 2] += np.random.randn(n_samples) * 0.5
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
    if "Malignant" in prediction or "Cancer Detected" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    elif "Suspicious" in prediction:
        recommendation = "Seek professional medical consultation for confirmation."
    else:
        recommendation = "Routine medical checkups are advised."
    result = f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"
    return result

# ==================================================
# Gemini AI Explanation (Fixed)
# ==================================================
def generate_gemini_explanation(prediction_text):
    """
    Generates patient-friendly explanation using Gemini AI
    """
    if not genai.api_key:
        return "Gemini AI key not set. Explanation unavailable."

    prompt = f"Explain this medical result to a patient in simple terms:\n{prediction_text}"
    
    try:
        response = genai.ChatCompletion.create(
            model="chat-bison-001",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response.choices[0].message.content
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

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
    gemini_explanation = generate_gemini_explanation(prediction_text)

    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=prediction_text,
        gemini_explanation=gemini_explanation
    )

# ==================================================
# Keep all other routes exactly the same (lung, prostate, clustering, self-test, API, health)
# ==================================================
# Copy all previous lung/prostate/selftest/clustering/api/health routes here without change

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    print("🚀 SmartOnco System Started")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
