import pandas as pd
import matplotlib.pyplot as plt

# Load the cleansed dataset
df = pd.read_csv('M:/Deakin Assignments/ML/PQC_data_cleansed.csv')

# 1. Distribution of Game Types
plt.figure(figsize=(8, 6))
df['Game_Type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Game Types')
plt.xlabel('Game Type')
plt.ylabel('Count')
plt.show()

# 2. Distribution of Release Years
plt.figure(figsize=(12, 6))
df['Released_Year'].dropna().hist(bins=30, color='purple')
plt.title('Distribution of Game Release Years')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()

# 3. Distribution of Age Categories
plt.figure(figsize=(8, 6))
df['Age_Category'].value_counts().plot(kind='bar', color='orange')
plt.title('Distribution of Age Categories')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()

# 4. Distribution of Minimum Number of Players Required
plt.figure(figsize=(8, 6))
df['Min_Players'].value_counts().plot(kind='bar', color='green')
plt.title('Distribution of Minimum Number of Players Required')
plt.xlabel('Minimum Number of Players')
plt.ylabel('Count')
plt.show()

# 5. Distribution of Maximum Number of Players Allowed
plt.figure(figsize=(8, 6))
df['Max_Players'].value_counts().plot(kind='bar', color='red')
plt.title('Distribution of Maximum Number of Players Allowed')
plt.xlabel('Maximum Number of Players')
plt.ylabel('Count')
plt.show()

# 6. Distribution of Average Play Time
plt.figure(figsize=(10, 6))
df['Average_Play_Time'].plot(kind='hist', bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Average Play Time')
plt.xlabel('Average Play Time (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 7. Scatter plot: Average Play Time vs Average Rating
plt.figure(figsize=(10, 6))
plt.scatter(df['Average_Play_Time'], df['Average_Rating'], alpha=0.5, color='blue')
plt.title('Relationship Between Average Play Time and Average Game Rating')
plt.xlabel('Average Play Time (minutes)')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# 8. Scatter plot: Average Complexity vs Average Rating
plt.figure(figsize=(10, 6))
plt.scatter(df['Average_Complexity'], df['Average_Rating'], alpha=0.5, color='green')
plt.title('Relationship Between Game Complexity and Average Game Rating')
plt.xlabel('Average Complexity')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Heatmap of Game Features')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()
