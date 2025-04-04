import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\Mohan\Desktop\Re admissions\re_admissions.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()
print("Updated Columns:", df.columns)  # Debugging step

# Ensure 'readmitted' column exists
if 'readmitted' not in df.columns:
    raise KeyError("Column 'readmitted' not found. Check dataset structure.")

# Prepare features and target variable
X = df.drop(columns=['readmitted'])
y = df['readmitted']

# Handle categorical variables (if needed)
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save results
with open("model_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(classification_report(y_test, y_pred))
