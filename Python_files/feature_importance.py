import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv(r'C:\Users\Mohan\Desktop\Re admissions\re_admissions.csv')

# Check column types and see which need encoding
print(df.dtypes)

# Handle missing values by filling them (you can use a default value or strategy)
binary_columns = ['glucose_test', 'A1Ctest', 'change_1', 'diabetes_med']  # Modify this list as necessary

# Fill missing values with 'no' for binary columns, or use a different strategy if needed
for col in binary_columns:
    df[col].fillna('no', inplace=True)
    df[col] = df[col].map({'yes': 1, 'no': 0}).astype(int)

# Encode other categorical variables using LabelEncoder
categorical_columns = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3']  # Add any other categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Convert age ranges (if necessary) to numeric values
def convert_age_range_to_numeric(age_range):
    if isinstance(age_range, str) and age_range.startswith('['):
        return int(age_range.split('-')[0][1:])
    return age_range

df['age'] = df['age'].apply(convert_age_range_to_numeric)

# Now, drop the target column ('readmitted') and assign the features (X) and target (y)
X = df.drop(columns=["readmitted"])
y = df["readmitted"]

# Train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]
top_features = features[sorted_indices][:10]  # Top 10 features

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(top_features, importances[sorted_indices][:10])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Features Influencing Readmission")
plt.show()
