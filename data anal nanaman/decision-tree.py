import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
file_path = "shopzada_churn.csv"  # Update this with your file path
data = pd.read_csv(file_path)

# Convert categorical variables to numerical labels
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split data into features and target variable
X = data.drop('Target_Churn', axis=1)  # Features
y = data['Target_Churn']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Generate rules
tree_rules = export_text(clf, feature_names=list(X.columns))

# Output the rules
print("Decision Tree Rules:")
print(tree_rules)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Churn', 'Churn'])
plt.show()
