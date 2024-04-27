import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
housePriceData = pd.read_csv('data/housePriceData.csv')


# Define Unseen Data
def set_aside_unseen_data(housePriceData, percent=0.1):
    unseen_data, remaining_data = train_test_split(housePriceData, test_size=percent, random_state=42)
    return unseen_data, remaining_data


# Define Training Data and Testing Data
def split_training_testing_data(housePriceData, percent_train=0.8):
    training_data, testing_data = train_test_split(housePriceData, test_size=1 - percent_train, random_state=42)
    return training_data, testing_data


# Separate columns for SLR and MLR
slr_data = housePriceData[['damages', 'discount']]
mlr_data = housePriceData[
    ['size', 'bedrooms', 'bathrooms', 'extraRooms', 'garage',
     'garden', 'inSubdivision', 'inCity', 'solarPowered',
     'price']]

# Set aside 10% of the dataset for unseen data
slr_unseen_data, slr_remaining_data = set_aside_unseen_data(slr_data)
mlr_unseen_data, mlr_remaining_data = set_aside_unseen_data(mlr_data)

# Split the remaining data into training and testing sets
slr_train, slr_test = split_training_testing_data(slr_remaining_data)
mlr_train, mlr_test = split_training_testing_data(mlr_remaining_data)

# Simple Linear Regression
# a. Display descriptive statistics of discount amounts
print("Descriptive statistics of discount amounts:")
print(slr_train['discount'].describe())

# b. Scatter plot: damages vs discount
plt.scatter(slr_train['damages'], slr_train['discount'])
plt.title('Damages vs Discount')
plt.xlabel('Damages')
plt.ylabel('Discount')
plt.show()

# c. Correlation between damages and discount
correlation = slr_train['damages'].corr(slr_train['discount'])
print("Correlation between damages and discount:", correlation)

# d. Create SLR model
slr_model = LinearRegression()
slr_model.fit(slr_train[['damages']], slr_train['discount'])
print("Simple Linear Regression Model:")
print("Intercept:", slr_model.intercept_)
print("Coefficient:", slr_model.coef_)

# e. Evaluate SLR model
slr_pred = slr_model.predict(slr_test[['damages']])
slr_rmse = np.sqrt(mean_squared_error(slr_test['discount'], slr_pred))
slr_r2 = r2_score(slr_test['discount'], slr_pred)
print("SLR RMSE:", slr_rmse)
print("SLR R^2:", slr_r2)

# f. Predict discount on unseen data
slr_unseen_pred = slr_model.predict(slr_unseen_data[['damages']])
print("Predicted discount on unseen data:", slr_unseen_pred)

# Multiple Linear Regression
# a. Display descriptive statistics of selling price
print("Descriptive statistics of selling price:")
print(mlr_train['price'].describe())

# b. Create MLR model
mlr_features = ['size', 'bedrooms', 'bathrooms', 'extraRooms', 'garage',
                'garden', 'inSubdivision', 'inCity', 'solarPowered']
mlr_model = LinearRegression()
mlr_model.fit(mlr_train[mlr_features], mlr_train['price'])
print("Multiple Linear Regression Model:")
print("Intercept:", mlr_model.intercept_)
print("Coefficients:", mlr_model.coef_)

# c. Evaluate MLR model
mlr_pred = mlr_model.predict(mlr_test[mlr_features])
mlr_rmse = np.sqrt(mean_squared_error(mlr_test['price'], mlr_pred))
mlr_r2 = r2_score(mlr_test['price'], mlr_pred)
print("MLR RMSE:", mlr_rmse)
print("MLR R^2:", mlr_r2)

# d. Predict house prices on unseen data
mlr_unseen_pred = mlr_model.predict(mlr_unseen_data[mlr_features])
print("Predicted house prices on unseen data:", mlr_unseen_pred)
