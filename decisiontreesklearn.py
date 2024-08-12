import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'M:/Deakin Assignments/ML/PQC_data_cleansed.csv'
df = pd.read_csv(file_path)

# 1. Feature Selection
features = df[['Average_Play_Time', 'Average_Complexity', 'Owner_Number', 
               'Trader_Number', 'HighInterest_Number', 'Interest_Number']]
target = df['Average_Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the decision tree with max_depth=6
tree = DecisionTreeRegressor(max_depth=6, random_state=42)
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared (R²): {r2}")

# Display the first 5 predicted values and compare them with the actual values
print("\nPredicted vs Actual values for Average Rating (first 5 examples):")
for i in range(5):
    print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")
