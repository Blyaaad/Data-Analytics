import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/shopzada_churn.csv"  # Update this with your file path
data = pd.read_csv(file_path)

# Assuming the target column is named 'Target_Churn', replace it with the actual name of your target column
X = data.drop('Target_Churn', axis=1)  # Features
y = data['Target_Churn']  # Target variable

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X)

# Split data into training (70%), testing (20%), and unseen (10%) datasets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_unseen, y_test, y_unseen = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

# Create a Naive Bayes Classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Predictions
y_pred_train = nb_classifier.predict(X_train)
y_pred_test = nb_classifier.predict(X_test)
y_pred_unseen = nb_classifier.predict(X_unseen)

# Model Evaluation
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_unseen = accuracy_score(y_unseen, y_pred_unseen)

precision_train = precision_score(y_train, y_pred_train)
precision_test = precision_score(y_test, y_pred_test)
precision_unseen = precision_score(y_unseen, y_pred_unseen)

recall_train = recall_score(y_train, y_pred_train)
recall_test = recall_score(y_test, y_pred_test)
recall_unseen = recall_score(y_unseen, y_pred_unseen)

roc_auc_train = roc_auc_score(y_train, y_pred_train)
roc_auc_test = roc_auc_score(y_test, y_pred_test)
roc_auc_unseen = roc_auc_score(y_unseen, y_pred_unseen)

# Confusion Matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

# Confusion Matrix for testing data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Confusion Matrix for unseen data
conf_matrix_unseen = confusion_matrix(y_unseen, y_pred_unseen)

# Plot ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_train)
roc_auc_curve_train = auc(fpr_train, tpr_train)

plt.figure()
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='ROC curve - Training Data (area = %0.2f)' % roc_auc_curve_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive-Bayes ROC Curve - Training Data')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve for testing data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
roc_auc_curve_test = auc(fpr_test, tpr_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve - Testing Data (area = %0.2f)' % roc_auc_curve_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive-Bayes ROC Curve - Testing Data')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve for unseen data
fpr_unseen, tpr_unseen, thresholds_unseen = roc_curve(y_unseen, y_pred_unseen)
roc_auc_curve_unseen = auc(fpr_unseen, tpr_unseen)

plt.figure()
plt.plot(fpr_unseen, tpr_unseen, color='green', lw=2, label='ROC curve - Unseen Data (area = %0.2f)' % roc_auc_curve_unseen)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive-Bayes ROC Curve - Unseen Data')
plt.legend(loc="lower right")
plt.show()

# Display results
print("Results for Training Data:")
print("Naive-Bayes")
print("Confusion Matrix:\n", conf_matrix_train)
print("Accuracy:", accuracy_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("ROC-AUC:", roc_auc_train)

print("\nResults for Testing Data:")
print("Naive-Bayes")
print("Confusion Matrix:\n", conf_matrix_test)
print("Accuracy:", accuracy_test)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("ROC-AUC:", roc_auc_test)

print("\nResults for Unseen Data:")
print("Naive-Bayes")
print("Confusion Matrix:\n", conf_matrix_unseen)
print("Accuracy:", accuracy_unseen)
print("Precision:", precision_unseen)
print("Recall:", recall_unseen)
print("ROC-AUC:", roc_auc_unseen)
