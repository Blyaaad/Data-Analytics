"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
model.fit(X_train, y_train)

# Make predictions on the unseen data
unseen_predictions = model.predict(X_unseen)

# Calculate R-squared score on the test set
test_r2_score = model.score(X_test, y_test)
print("R-squared score on the test set:", test_r2_score)

# Display coefficients
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)"""

#ITO LANG ILALAGAY SA TABLE B
# Create scatter plot
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted line')
plt.xlabel('Number of Damages')
plt.ylabel('Discount Amount')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
