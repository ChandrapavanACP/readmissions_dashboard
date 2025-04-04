import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load dataset (ensure to update the path)
df = pd.read_csv(r'C:\Users\Mohan\Desktop\Re admissions\re_admissions.csv')

# Handle missing values for binary and categorical columns as shown earlier
binary_columns = ['glucose_test', 'A1Ctest', 'change_1', 'diabetes_med']  # Modify this list as necessary
for col in binary_columns:
    df[col].fillna('no', inplace=True)
    df[col] = df[col].map({'yes': 1, 'no': 0}).astype(int)

categorical_columns = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']  # Add any other categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Convert age ranges to numeric if necessary
def convert_age_range_to_numeric(age_range):
    if isinstance(age_range, str) and age_range.startswith('['):
        return int(age_range.split('-')[0][1:])
    return age_range

df['age'] = df['age'].apply(convert_age_range_to_numeric)

# Define feature and target variables
X = df.drop(columns=["readmitted"])
y = df["readmitted"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get probability predictions for the positive class
y_probs = rf.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Readmission Prediction")
plt.legend()
plt.show()
