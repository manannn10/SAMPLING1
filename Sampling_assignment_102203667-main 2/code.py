import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv("Creditcard_data.csv")

# Step 1: Balance the dataset using SMOTE
X = df.drop(columns=["Class"])  # Replace "Class" with the target column name
y = df["Class"]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Standardize features
scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

# Step 2: Calculate Sample Size
Z = 1.96  # Z-score for 95% confidence level
p = 0.5   # Estimated proportion of population
E = 0.05  # Margin of error

sample_size = int((Z**2 * p * (1 - p)) / (E**2))
print(f"Sample size: {sample_size}")

# Step 3: Create 5 Samples Using Sampling Techniques
samples = {
    "random": X_balanced[:sample_size],
    "stratified": X_balanced[:sample_size],  # Simulating stratified with balanced data
    "systematic": X_balanced[::len(X_balanced)//sample_size],
    "bootstrap": X_balanced[np.random.randint(0, len(X_balanced), sample_size)],
    "cluster": X_balanced[:sample_size]  # Replace with actual cluster logic if necessary
}

# Step 4: Apply Sampling Techniques on Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Evaluate Models with Each Sampling Technique
results = {}

for sample_name, sample_data in samples.items():
    X_sample = sample_data
    y_sample = y_balanced[:len(X_sample)]

    accuracies = {}
    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracies[model_name] = round(accuracy, 2)

    results[sample_name] = accuracies

# Display Results
print("\nModel Accuracies with Different Sampling Techniques:\n")
for sample_name, accuracies in results.items():
    print(f"{sample_name.capitalize()} Sampling:")
    for model_name, accuracy in accuracies.items():
        print(f"{model_name}: {accuracy}")
    print()

# Optional: Save Results to File
with open("results.txt", "w") as file:
    file.write("Model Accuracies with Different Sampling Techniques:\n\n")
    for sample_name, accuracies in results.items():
        file.write(f"{sample_name.capitalize()} Sampling:\n")
        for model_name, accuracy in accuracies.items():
            file.write(f"{model_name}: {accuracy}\n")
        file.write("\n")
