import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data Collection
# Load the dataset from Kaggle
data = pd.read_csv('data/heart_failure_clinical_records.csv')



# Calculate percentage of patients with heart failure and without heart failure
total_instances = len(data)
heart_failure_count = data['DEATH_EVENT'].sum()
no_heart_failure_count = total_instances - heart_failure_count
heart_failure_by_gender = data.groupby('sex')['DEATH_EVENT'].mean() * 100

percentage_heart_failure = (heart_failure_count / total_instances) * 100
percentage_no_heart_failure = (no_heart_failure_count / total_instances) * 100

heart_failure_instances = data[data['DEATH_EVENT'] == 1].shape[0]

categories = ['Heart Failure', 'No Heart Failure']
percentages = [percentage_heart_failure, percentage_no_heart_failure]

age_bins = np.arange(40, 90, 5)
age_groups = pd.cut(data['age'], bins=age_bins)
age_groups_counts = data.groupby([age_groups, 'DEATH_EVENT']).size().unstack(fill_value=0)


print("Number of rows in the dataset:", total_instances)
print("Instances of patients diagnosed with heart failure:", heart_failure_instances)
print("Percentage of patients with heart failure:", percentage_heart_failure)
print("Number of patients without heart failure:", no_heart_failure_count)
print("Percentage of patients without heart failure:", percentage_no_heart_failure)

# Create bar plot for Heart Failure Status
plt.figure(figsize=(8, 6))
plt.bar(categories, percentages, color=['red', 'blue'])
plt.title('Percentage of Patients with and without Heart Failure')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()

# Create bar plot for Sex
plt.figure(figsize=(8, 6))
heart_failure_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title('Percentage of Patients with Heart Failure by Gender')
plt.xlabel('Gender (0: Female, 1: Male)')
plt.ylabel('Percentage of Patients with Heart Failure')
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.show()

# Create bar plot for Age
plt.figure(figsize=(12, 6))
age_groups_counts.plot(kind='bar', stacked=True, color=['green', 'orange'])
plt.title('Distribution of Heart Failure by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Heart Failure', labels=['No', 'Yes'])
plt.show()

# Data Preprocessing
# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Data Analysis
# Correlation Matrix with Heatmap
correlation_matrix = data_imputed.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Define features and target variable
X = data_imputed.drop(columns=['DEATH_EVENT'])
y = data_imputed['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize machine learning models
models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier()
}

# Data Training and Testing
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # Data Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)  # Handle zero division
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

    # Print evaluation metrics
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

# Hyperparameter Tuning Example with GridSearchCV (for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Evaluate the best model from GridSearchCV
y_pred_best_rf = best_rf.predict(X_test)
y_proba_best_rf = best_rf.predict_proba(X_test)[:, 1]

accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
precision_best_rf = precision_score(y_test, y_pred_best_rf, zero_division=1)  # Handle zero division
recall_best_rf = recall_score(y_test, y_pred_best_rf)
f1_best_rf = f1_score(y_test, y_pred_best_rf)
roc_auc_best_rf = roc_auc_score(y_test, y_proba_best_rf)

print("Best Random Forest Model after Hyperparameter Tuning")
print(f"Accuracy: {accuracy_best_rf:.2f}")
print(f"Precision: {precision_best_rf:.2f}")
print(f"Recall: {recall_best_rf:.2f}")
print(f"F1 Score: {f1_best_rf:.2f}")
print(f"ROC AUC: {roc_auc_best_rf:.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best_rf)}")
