import pandas as pd
import numpy as np

# Load the dataset
file_path = 'M:/Deakin Assignments/ML/PQC_data_cleansed.csv'
df = pd.read_csv(file_path)

# 1. Feature Engineering - Adding Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

# Select the features and the target variable
features = df[['Average_Play_Time', 'Average_Complexity', 'Owner_Number', 
               'Trader_Number', 'HighInterest_Number', 'Interest_Number']].values
target = df['Average_Rating'].values

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(features)

# 2. Add a column of ones to the features to account for the intercept term
X = np.c_[np.ones(X_poly.shape[0]), X_poly]

# 3. Split the data into training and testing sets manually (70% train, 30% test)
split_index = int(0.7 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = target[:split_index], target[split_index:]

# 4. Implement the Linear Regression using the Normal Equation
# Normal Equation: theta = (X.T * X)^(-1) * X.T * y
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# 5. Make predictions on the test set
y_pred = X_test @ theta

# 6. Evaluate the model
# Mean Squared Error (MSE)
mse = np.mean((y_pred - y_test) ** 2)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R²)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

# Display the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared (R²): {r2}")

# Display the predicted values for the first 5 examples
print("\nPredicted values for Average Rating (first 5 examples):")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test[i]:.2f}")
