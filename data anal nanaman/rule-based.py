from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import wittgenstein as lw

# Read the dataset with specified column names
df = pd.read_csv('shopzada_churn.csv', usecols=['Customer_ID', 'Age', 'Gender', 'Annual_Income', 'Total_Spend',
                                                 'Years_as_Customer', 'Num_of_Purchases', 'Average_Transaction_Amount',
                                                 'Num_of_Returns', 'Num_of_Support_Contacts', 'Satisfaction_Score',
                                                 'Last_Purchase_Days_Ago', 'Email_Opt_In', 'Promotion_Response',
                                                 'Target_Churn'])

# Check if there are positive samples for 'Target_Churn'
if df['Target_Churn'].nunique() < 2:
    print("There are not enough positive samples in the dataset.")
    exit()

# Split the dataset into train and test sets
train, test = train_test_split(df, test_size=.7, random_state=42)

# Check if there are positive samples in the training set
if not train['Target_Churn'].any():
    print("There are no positive samples in the training set.")
    exit()

# Initialize and train the classifier with 'Target_Churn' as the target variable
clf = lw.RIPPER(random_state=42)
try:
    clf.fit(train, class_feat='Target_Churn', pos_class='Responded')
except ValueError as e:
    print("Error:", e)
    exit()

# Prepare test data
X_test = test.drop('Target_Churn', axis=1)
y_test = test['Target_Churn']

# Calculate precision and recall scores
try:
    precision = precision_score(y_test, clf.predict(X_test), zero_division=0)
    recall = recall_score(y_test, clf.predict(X_test), zero_division=0)
    print(f'precision: {precision} recall: {recall}')
except ValueError as e:
    print("Error calculating precision and recall scores:", e)
    exit()

# Output rules
try:
    clf.ruleset_.out_pretty()
except AttributeError as e:
    print("Error outputting rules:", e)
