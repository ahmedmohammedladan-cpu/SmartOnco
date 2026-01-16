from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
# Gemini AI Configuration
# =========================
# Get API key from Render environment variable
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")

# Configure Gemini
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    print("Gemini API configured successfully")
else:
    print("Warning: GENAI_API_KEY environment variable not set")

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
    """Generate explanation when Gemini is unavailable"""
    if "Malignant" in prediction_text:
        if "High" in prediction_text:
            return "🚨 **High Risk Alert**: This indicates a strong likelihood of cancer cells. Immediate consultation with an oncologist is crucial. They will likely recommend further tests like biopsy, MRI, or additional imaging to confirm and determine the exact type and stage."
        elif "Moderate" in prediction_text:
            return "⚠️ **Moderate Risk**: This suggests possible cancer cells that require medical attention. Schedule an appointment with an oncologist for further evaluation. Additional testing may be needed to confirm the diagnosis and plan appropriate treatment."
        else:
            return "🔍 **Further Evaluation Needed**: This result suggests cancer cells may be present. An oncologist can provide comprehensive evaluation, discuss treatment options, and create a personalized care plan."
    else:
        if "Low" in prediction_text:
            return "✅ **Low Risk - Benign**: This indicates non-cancerous characteristics. Continue with regular screenings as recommended by your doctor. Maintain breast health through monthly self-exams and annual clinical checkups."
        else:
            return "✅ **Benign Result**: The analysis shows no cancer cells detected. Continue with routine medical checkups. Regular monitoring is key to maintaining breast health."

# ==================================================
# Gemini AI Explanation
# ==================================================
def generate_gemini_explanation(prediction_text):
    """
    Generates patient-friendly explanation using Gemini AI
    """
    if not GENAI_API_KEY:
        return "🔧 **Setup Required**: Gemini API key not configured. " + generate_fallback_explanation(prediction_text)
    
    prompt = f"""You are a compassionate medical assistant. Explain this breast cancer screening result to a patient in simple, clear, and reassuring language:

RESULT: {prediction_text}

Please provide:
1. A simple explanation of what this means
2. What the risk level indicates
3. Why the recommendation is given
4. Next steps in plain language
5. A reassuring, hopeful tone

Keep it under 150 words, avoid medical jargon, and be empathetic."""

    try:
        # Use gemini-1.5-flash-latest for latest model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                max_output_tokens=200,
            )
        )
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Error: {error_msg}")  # This will appear in Render logs
        
        # Check for specific error types
        if "quota" in error_msg.lower():
            return "📊 **API Limit**: Gemini API quota exceeded. " + generate_fallback_explanation(prediction_text)
        elif "permission" in error_msg.lower() or "403" in error_msg:
            return "⚠️ **API Access Issue**: Please verify Generative Language API is enabled in Google Cloud Console. " + generate_fallback_explanation(prediction_text)
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return "⏳ **Rate Limited**: Too many requests. Please try again in a moment. " + generate_fallback_explanation(prediction_text)
        else:
            return "🤖 **AI Service Temporarily Unavailable**: " + generate_fallback_explanation(prediction_text)

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
        feature_vals = [
            float(request.form.get(f"feature_bc{i+1}", 0))
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
        
    except ValueError as e:
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text="Error: Please enter valid numeric values for all features.",
            gemini_explanation="All input fields must contain numbers. Please check your entries and try again."
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text="An unexpected error occurred. Please try again.",
            gemini_explanation="System error. Please ensure all inputs are correctly filled and try again."
        )

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
