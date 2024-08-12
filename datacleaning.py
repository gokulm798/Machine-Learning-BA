import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'M:/Deakin Assignments/ML/PQC_data.csv'
df = pd.read_csv(file_path)

# Check and display missing values before cleansing
missing_values_before = df.isnull().sum()
print("Missing Values Before Cleansing:")
print(missing_values_before)

# 1. Handle missing values without using inplace=True
df['Game_Name'] = df['Game_Name'].fillna('Unknown')

# 2. Check for duplicates
duplicates = df.duplicated().sum()

# 3. Handle special placeholder values
median_year = df['Released_Year'].median()
df['Released_Year'] = df['Released_Year'].replace(-99, median_year)

# 4. Detect and Display Outliers in Average_Play_Time
# Calculate Q1 (25th percentile) and Q3 (75th percentile) for Average_Play_Time
Q1 = df['Average_Play_Time'].quantile(0.25)
Q3 = df['Average_Play_Time'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['Average_Play_Time'] < lower_bound) | (df['Average_Play_Time'] > upper_bound)]
outliers_count = outliers.shape[0]

# Display the outliers
print("Outliers detected in 'Average_Play_Time':")
print(outliers[['Game_Name', 'Average_Play_Time']])

# Plot the box-and-whisker plot before handling outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Average_Play_Time'])
plt.title('Box-and-Whisker Plot of Average Play Time (Before Outlier Handling)')
plt.xlabel('Average Play Time')
plt.show()

# Handle outliers: Replace outliers with the median value
median_value = df['Average_Play_Time'].median()
df['Average_Play_Time'] = df['Average_Play_Time'].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)

# Plot the box-and-whisker plot after handling outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Average_Play_Time'])
plt.title('Box-and-Whisker Plot of Average Play Time (After Outlier Handling)')
plt.xlabel('Average Play Time')
plt.show()

# Re-check for any remaining missing values after the cleansing process
remaining_missing_values = df.isnull().sum()

# Saving the cleansed dataset to a new CSV file
cleansed_file_path = 'M:/Deakin Assignments/ML/PQC_data_cleansed.csv'
df.to_csv(cleansed_file_path, index=False)

# Display the results of data cleansing
cleansing_summary = {
    "Missing Values After Cleansing": remaining_missing_values,
    "Number of Duplicate Rows": duplicates,
    "Number of Outliers Detected and Replaced": outliers_count,
}

# Display the summary of cleansing
print("\nCleansing Summary:")
for key, value in cleansing_summary.items():
    print(f"{key}: {value}")

# Path to the cleansed data file
print(f"Cleansed data saved to: {cleansed_file_path}")
