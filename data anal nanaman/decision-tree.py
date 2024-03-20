import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the CSV file
file_path = "shopzada_churn.csv"
data = pd.read_csv(file_path)

# Assuming the target column is named 'Target_Churn'
X = data.drop(['Customer_ID', 'Age', 'Gender', 'Annual_Income', 'Total_Spend', 'Years_as_Customer', 'Num_of_Purchases',
               'Average_Transaction_Amount',
               'Num_of_Returns', 'Num_of_Support_Contacts', 'Satisfaction_Score', 'Last_Purchase_Days_Ago',
               'Email_Opt_In', 'Promotion_Response', 'Target_Churn'], axis=1)  # Features
y = data['Target_Churn']  # Target variable

# Check the first few rows of the DataFrame to ensure data loading was successful
print(data.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
