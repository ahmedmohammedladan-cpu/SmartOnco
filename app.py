from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

# =========================
# Gemini AI Import - with fallback
# =========================
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI not installed. AI explanations will use fallback.")

# =========================
# Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# Gemini AI Configuration
# =========================
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")

if GENAI_AVAILABLE and GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        GEMINI_ENABLED = True
        print("Gemini API configured")
    except:
        GEMINI_ENABLED = False
else:
    GEMINI_ENABLED = False

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
# RESULT FORMATTING
# ==================================================
def format_result(prediction, risk):
    if "Malignant" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    else:
        recommendation = "Routine medical checkups are advised."
    return f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"

# ==================================================
# Fallback Explanation
# ==================================================
def generate_fallback_explanation(prediction_text):
    if "Malignant" in prediction_text:
        return "This indicates possible cancer cells. Please consult with an oncologist for further testing and evaluation."
    else:
        return "This suggests benign characteristics. Continue with regular checkups as recommended."

# ==================================================
# Gemini AI Explanation
# ==================================================
def generate_gemini_explanation(prediction_text):
    if not GEMINI_ENABLED:
        return generate_fallback_explanation(prediction_text)
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Explain this in simple terms: {prediction_text}")
        return response.text
    except:
        return generate_fallback_explanation(prediction_text)

# ==================================================
# WEB ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=None,
        gemini_explanation=None
    )

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
        feature_vals = [float(request.form.get(f"feature_bc{i+1}", 0)) 
                       for i in range(len(selected_features_bc))]
        
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
    except Exception as e:
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text=f"Error: {str(e)}",
            gemini_explanation="Please check your inputs and try again."
        )

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
