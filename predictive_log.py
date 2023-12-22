import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.preprocessing import StandardScaler

path = 'data.csv'
df = pd.read_csv(path)

fico_Z = stats.zscore(df['FICO_score'])
debt_Z = stats.zscore(df['Debt_To_Income_Less_Housing'])

# Set a threshold for Z-scores (e.g., 3 standard deviations)
threshold = 3
t = 4

# Identify outliers based on the threshold
fico_outlier = abs(fico_Z) > threshold
debt_outlier = abs(debt_Z) > t

# Remove outliers from the DataFrame
df = df[~fico_outlier]
df = df[~debt_outlier]



# Define features and target variable
X = df[['FICO_score', 'Debt_To_Income_Less_Housing']]
y = df['Approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)

# Fit the model on the resampled training data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Change the threshold (e.g., 0.3)
custom_threshold = 0.3
y_pred_custom_threshold = (y_prob > custom_threshold).astype(int)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
