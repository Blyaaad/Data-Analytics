import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# Load the data
data = pd.read_csv("data/heart_failure_clinical_records.csv")

# Data Exploration
print("Data Exploration:")
print(data.head())
print(data.info())
print(data.describe())

# Data Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Split features and target variable
X = data.drop(columns=['DEATH_EVENT'])
y = data['DEATH_EVENT']

# Feature Importance
model = RandomForestClassifier()
model.fit(X, y)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importance:")
print(feature_importance.sort_values(ascending=False))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Model Interpretation using SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)
shap.summary_plot(shap_values, X, plot_type="bar")