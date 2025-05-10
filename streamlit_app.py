# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
#url = 'FatalityCrashData.csv.csv'  # Replace with the path or URL to your dataset
data = pd.read_csv('FatalityCrashData.csv')

# Display first few rows of the dataset
print(data.head())

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values (depending on the dataset)
data = data.dropna()  # or data.fillna(method='ffill')

# Convert 'CrashDateTime' to datetime format
data['CrashDateTime'] = pd.to_datetime(data['CrashDateTime'])

# Feature Engineering (example: extracting day, month, hour from 'CrashDateTime')
data['Year'] = data['CrashDateTime'].dt.year
data['Month'] = data['CrashDateTime'].dt.month
data['Day'] = data['CrashDateTime'].dt.day
data['Hour'] = data['CrashDateTime'].dt.hour

# Selecting relevant features for prediction (you can add more features as per your project)
features = ['Fatals', 'Peds', 'Persons', 'TotalVehicles', 'Year', 'Month', 'Day', 'Hour']
X = data[features]
y = data['Fatals']  # Target: Fatals or any other column you want to predict

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model - Random Forest Classifier (you can choose another model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualizations
# Plot feature importance
features_importance = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, features_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# You can further create additional visualizations (e.g., accident trends over time, etc.)
