# fix_prostate_model.py
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def fix_prostate_model():
    print("🔄 Fixing prostate cancer model with safe medical parameters...")
    
    # Load your existing prostate cancer data
    df = pd.read_csv('prostate_cancer.csv')
    
    # Identify features and target (adjust based on your data structure)
    # Assuming the last column is the target like in most medical datasets
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column as target
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    # Safe medical parameters - FIXING THE OVERFITTING
    safe_model = DecisionTreeClassifier(
        max_depth=8,              # Was: None (infinite)
        min_samples_leaf=10,      # Was: 1 (dangerous)
        min_samples_split=20,     # Was: 2 (too sensitive)
        random_state=42
    )

    # Retrain with safe parameters
    safe_model.fit(X, y)
    
    # Save the fixed model (replace the problematic one)
    with open('prostate_cancer_model_FIXED.pkl', 'wb') as f:
        pickle.dump(safe_model, f)
    
    print("✅ Model fixed and saved as 'prostate_cancer_model_FIXED.pkl'")
    
    # Compare with old model
    train_score = safe_model.score(X, y)
    print(f"✅ Fixed model training accuracy: {train_score:.1%}")
    
    # Show the most important medical features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': safe_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top medical features driving predictions:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    fix_prostate_model()
