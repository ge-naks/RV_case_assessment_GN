import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats

# Load the data
path = 'data.csv'
df = pd.read_csv(path)

# Drop non-numeric columns and target variable
X = df.drop(['User ID', 'Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender', 'Approved', 'bounty', 'Loan_Amount', 'Monthly_Housing_Payment', 'Monthly_Gross_Income', 'Ever_Bankrupt_or_Foreclose'], axis=1)
y = df['Approved']

# Identify and remove outliers
fico_Z = stats.zscore(df['FICO_score'])
debt_Z = stats.zscore(df['Debt_To_Income_Less_Housing'])
threshold = 3
t = 4
fico_outlier = abs(fico_Z) > threshold
debt_outlier = abs(debt_Z) > t
df = df[~fico_outlier]
df = df[~debt_outlier]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to oversample the minority class on the scaled data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)

# Fit the model on the resampled training data
model.fit(X_train_resampled, y_train_resampled)

# Get feature coefficients
feature_importance = pd.Series(model.coef_[0], index=X.columns)

# Sort features by their absolute coefficients
sorted_feature_importance = feature_importance.abs().sort_values(ascending=False)

# Print or visualize the sorted feature importance
print(sorted_feature_importance)