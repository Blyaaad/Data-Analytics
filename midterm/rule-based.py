import wittgenstein as lw
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the dataset with specified column names
df = pd.read_csv('data/shopzada_churn.csv')

# Encoding categorical variables
data_encoded = pd.get_dummies(df, columns=['Gender', 'Promotion_Response'])

# Splitting the data into features and target variable
X = data_encoded.drop(columns=['Customer_ID', 'Target_Churn'])
y = data_encoded['Target_Churn']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of X_train and X_test
print(X_train.shape, X_test.shape)

# Check if there are positive samples in the training set
print(y.value_counts())

threshold = 3
rule_based_pred = X_test['Satisfaction_Score'] >= threshold

print("\nRule-based Classifier:")
print("Accuracy:", accuracy_score(y_test, rule_based_pred))
print("Precision:", precision_score(y_test, rule_based_pred))
print("Recall:", recall_score(y_test, rule_based_pred))
print("F1 Score:", f1_score(y_test, rule_based_pred))

# Fit method to train a RIPPER or IREP classifier, with 'Target_Churn' as the target variable
clf = lw.RIPPER()
print("Responded")
clf.fit(X_train, class_feat='Promotion_Response', pos_class='Responded')

# Output rules
clf.ruleset_.out_pretty()