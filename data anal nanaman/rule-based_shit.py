from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import wittgenstein as lw

df = pd.read_csv('shopzada_churn.csv')
train, test = train_test_split(df, test_size=.7, random_state=42)
clf = lw.RIPPER()

clf.fit(train, class_feat='Promotion_Response', pos_class='Responded', random_state=42)

X_test = test.drop('Promotion_Response', axis=1)
y_test = test['Promotion_Response']

precision = clf.score(X_test, y_test, precision_score)
recall = clf.score(X_test, y_test, recall_score)
print(f'precision: {precision} recall: {recall}')

clf.ruleset_.out_pretty()
