from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for sessions

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
# CLUSTERING SETUP
# ==================================================
def perform_clustering(data_type, n_clusters=2):
    """Perform K-means clustering and return results"""
    try:
        if data_type == "breast":
            data = X_bc
            feature_names = selected_features_bc
        else:
            # For other cancers, you would load appropriate data
            return {"error": f"Clustering not implemented for {data_type}"}
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                             c=cluster_labels, cmap='viridis', 
                             alpha=0.6, edgecolors='w', s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'{data_type.capitalize()} Cancer Data Clustering (K={n_clusters})')
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Get cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_data = data[cluster_indices]
            stats = {
                'cluster': i,
                'size': len(cluster_indices),
                'centroid': kmeans.cluster_centers_[i].tolist(),
                'mean_values': cluster_data.mean(axis=0).tolist() if len(cluster_indices) > 0 else []
            }
            cluster_stats.append(stats)
        
        return {
            "success": True,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": float(silhouette_avg),
            "plot_url": plot_url,
            "cluster_stats": cluster_stats,
            "n_clusters": n_clusters,
            "feature_names": feature_names
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# ==================================================
# MODEL SELF-TEST
# ==================================================
def perform_self_test(data_type):
    """Perform comprehensive model self-testing"""
    try:
        if data_type == "breast":
            model = model_bc
            X_test = X_test_bc
            y_test = y_test_bc
            X_train = X_train_bc
            y_train = y_train_bc
            feature_names = selected_features_bc
        elif data_type == "lung":
            # Load or create lung cancer dataset
            # For now, using dummy data
            return {"error": "Complete lung cancer dataset required for self-test"}
        elif data_type == "prostate":
            # Load or create prostate cancer dataset
            return {"error": "Complete prostate cancer dataset required for self-test"}
        else:
            return {"error": f"Unknown data type: {data_type}"}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        else:
            feature_importance = {}
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Create visualization for confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save plot
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        cm_plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return {
            "success": True,
            "accuracy": float(accuracy),
            "cross_val_scores": cv_scores.tolist(),
            "cross_val_mean": float(cv_scores.mean()),
            "cross_val_std": float(cv_scores.std()),
            "feature_importance": feature_importance,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "cm_plot_url": cm_plot_url,
            "model_type": str(type(model).__name__)
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

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
# CLUSTERING ROUTES
# ==================================================
@app.route("/cluster/<data_type>", methods=["GET", "POST"])
def cluster_data(data_type):
    if request.method == "POST":
        try:
            n_clusters = int(request.form.get("n_clusters", 2))
        except:
            n_clusters = 2
    else:
        n_clusters = 2
    
    result = perform_clustering(data_type, n_clusters)
    
    if request.headers.get('Content-Type') == 'application/json' or request.is_json:
        return jsonify(result)
    
    return render_template(f"{data_type}_clustering.html", result=result)

@app.route("/api/cluster/<data_type>", methods=["POST"])
def api_cluster_data(data_type):
    try:
        data = request.get_json()
        n_clusters = data.get("n_clusters", 2)
        result = perform_clustering(data_type, n_clusters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

# ==================================================
# SELF-TEST ROUTES
# ==================================================
@app.route("/selftest/<data_type>", methods=["GET"])
def self_test(data_type):
    result = perform_self_test(data_type)
    
    if request.headers.get('Content-Type') == 'application/json' or request.is_json:
        return jsonify(result)
    
    return render_template(f"{data_type}_selftest.html", result=result)

@app.route("/api/selftest/<data_type>", methods=["GET"])
def api_self_test(data_type):
    result = perform_self_test(data_type)
    return jsonify(result)

# ==================================================
# API ROUTES
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
            "recommendation": full_result.split(", ")[2].replace("Recommendation: ", ""),
            "full_result": full_result,
            "disclaimer": "This system is for decision support only and does not replace professional medical diagnosis."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

# ==================================================
# NEW ROUTES FOR CLUSTERING AND SELF-TEST PAGES
# ==================================================
@app.route("/breast/clustering")
def breast_clustering_page():
    return render_template("breast_clustering.html")

@app.route("/breast/selftest")
def breast_selftest_page():
    return render_template("breast_selftest.html")

@app.route("/lung/clustering")
def lung_clustering_page():
    return render_template("lung_clustering.html")

@app.route("/lung/selftest")
def lung_selftest_page():
    return render_template("lung_selftest.html")

@app.route("/prostate/clustering")
def prostate_clustering_page():
    return render_template("prostate_clustering.html")

@app.route("/prostate/selftest")
def prostate_selftest_page():
    return render_template("prostate_selftest.html")

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "breast_cancer": True,
            "lung_cancer": "manual_rules",
            "prostate_cancer": "manual_rules"
        }
    })

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
