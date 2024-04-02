"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the CSV file
file_path = 'housePriceData.csv'  # Replace 'your_file_path.csv' with the actual path to your CSV file
data = pd.read_csv(file_path)

# Extract relevant columns
X = data[['damages']].values
y = data['discount'].values

# Splitting 10% of the data for unseen data
X_train_all, X_unseen, y_train_all, y_unseen = train_test_split(X, y, test_size=0.10, random_state=42)

# Splitting the remaining data (90%) into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)"""

#ITO LANG ILALAGAY SA TABLE E
# Make predictions on the unseen data
unseen_predictions = model.predict(X_unseen)

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Evaluation metrics:")
print("Train Mean Squared Error:", train_mse)
print("Test Mean Squared Error:", test_mse)
print("Train Mean Absolute Error:", train_mae)
print("Test Mean Absolute Error:", test_mae)
print("Train R-squared Score:", train_r2)
print("Test R-squared Score:", test_r2)
