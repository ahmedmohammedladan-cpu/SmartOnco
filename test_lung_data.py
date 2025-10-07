import pandas as pd

data = pd.read_csv("survey lung cancer.csv")
print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
print("First 5 rows:")
print(data.head())
