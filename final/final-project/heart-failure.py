import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
import shap
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv("data/heart_failure_clinical_records.csv")

# Data Exploration
print("Data Exploration:")
print(data.head())
print(data.info())
print(data.describe())

# Impute missing values using kNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

# Verify imputation
print("Missing Values after kNN Imputation:")
print(data_imputed.isnull().sum())

# Data Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(data_imputed.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Split features and target variable
X = data_imputed.drop(columns=['DEATH_EVENT'])
y = data_imputed['DEATH_EVENT']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature Importance
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("Feature Importance:")
print(feature_importance.sort_values(ascending=False))

# Split data into train and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Model Interpretation using SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train, check_additivity=False)

# Ensure the summary plot uses the training data and corresponding SHAP values
shap.summary_plot(shap_values, X_train, feature_names=X.columns, max_display=None)