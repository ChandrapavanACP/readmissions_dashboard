import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv(r'C:\Users\Mohan\Desktop\Re admissions\re_admissions.csv')  # file path
# Define features (X) and target (y)
X = df.drop(columns=["id", "readmitted"])  # Drop 'id' and target variable
y = df["readmitted"]  # Target variable

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print dataset shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
