import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Multiple Linear Regression
print("----------------------------------------")
print("Multiple Linear Regression")

print()
# a. Display descriptive statistics of selling price
print("a. Display descriptive statistics of selling price")
print("Descriptive statistics of selling price:")
print(mlr_train['price'].describe())

print()
# b. Create MLR model
print("b. Create MLR model")
mlr_features = ['size', 'bedrooms', 'bathrooms', 'extraRooms', 'garage',
                'garden', 'inSubdivision', 'inCity', 'solarPowered']
mlr_model = LinearRegression()
mlr_model.fit(mlr_train[mlr_features], mlr_train['price'])
print("Multiple Linear Regression Model:")
print("Price =", mlr_model.intercept_, "\n+ \n (", mlr_model.coef_[0], " × size) \n+\n (", mlr_model.coef_[1], " × bedrooms) \n+ \n(", mlr_model.coef_[2], " × bathrooms) \n+\n (", mlr_model.coef_[3], " × extraRooms) \n+\n (", mlr_model.coef_[4], " × garage) \n+\n (", mlr_model.coef_[5], " × garden) \n+\n (", mlr_model.coef_[6], " × inSubdivision) \n+\n (", mlr_model.coef_[7], " × inCity) \n+\n (", mlr_model.coef_[8], " × solarPowered)\n", end="")


print()
# c. Evaluate MLR model
print("c. Evaluate MLR model")
mlr_pred = mlr_model.predict(mlr_test[mlr_features])

# Calculate evaluation metrics
mlr_rmse = np.sqrt(mean_squared_error(mlr_test['price'], mlr_pred))
mlr_r2 = r2_score(mlr_test['price'], mlr_pred)
mlr_mae = mean_absolute_error(mlr_test['price'], mlr_pred)
mlr_mse = mean_squared_error(mlr_test['price'], mlr_pred)

# Print evaluation metrics
print("MLR MAE:", mlr_mae)
print("MLR MSE:", mlr_mse)
print("MLR R^2:", mlr_r2)
print("MLR RMSE:", mlr_rmse)

print()
# d. Predict house prices on unseen data
print("d. Predict house prices on unseen data")
mlr_unseen_pred = mlr_model.predict(mlr_unseen_data[mlr_features])

actual_prices = pd.DataFrame({'Actual': mlr_unseen_data['price'].values, 'Predicted': mlr_unseen_pred})
print(actual_prices)