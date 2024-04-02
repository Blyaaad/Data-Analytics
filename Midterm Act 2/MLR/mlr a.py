"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('../SLR/housePriceData.csv')

# Selecting relevant columns
selected_columns = ['size', 'bedrooms', 'bathrooms', 'extraRooms', 'garage', 'garden', 'inSubdivision', 'inCity', 'solarPowered', 'price']
data = data[selected_columns]

# Splitting unseen data (10%)
train_data, unseen_data = train_test_split(data, test_size=0.1, random_state=42)

# Splitting training and testing data (80-20)
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('price', axis=1), train_data['price'], test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predicting on the unseen data
X_unseen = unseen_data.drop('price', axis=1)
y_unseen_true = unseen_data['price']
y_unseen_pred = model.predict(X_unseen)"""

#ITO LANG ILALAGAY SA MLR A TABLE
# Evaluating the model on unseen data
mse_unseen = mean_squared_error(y_unseen_true, y_unseen_pred)
print("Mean Squared Error on Unseen Data:", mse_unseen)

# Displaying descriptive statistics of the selling price amounts
print("Descriptive Statistics of Selling Price:")
print(data['price'].describe())
