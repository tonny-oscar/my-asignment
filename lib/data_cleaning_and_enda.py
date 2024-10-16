# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets from CSV files
df_transactions = pd.read_csv('transactions.csv')
df_products = pd.read_csv('products.csv')

# Display the first few rows of the datasets to understand their structure
print("Transactions Data:")
print(df_transactions.head())

print("\nProducts Data:")
print(df_products.head())

# Merge the two datasets using a common column (assuming it's 'product_id')
merged_df = pd.merge(df_transactions, df_products, on='product_id', how='inner')

# Check for missing values
print("\nMissing values in merged dataset:")
print(merged_df.isnull().sum())

# Handle missing values - Example: fill missing values with forward fill
merged_df.fillna(method='ffill', inplace=True)

# Remove duplicate rows
merged_df.drop_duplicates(inplace=True)

# Save the cleaned dataset to a new CSV file
merged_df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_data.csv'.")

# EDA: Conduct basic statistical analysis
print("\nBasic Statistical Description:")
print(merged_df.describe())

# Visualize sales distribution
plt.figure(figsize=(10,6))
sns.histplot(merged_df['sales'], kde=True, color='blue')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

#this is used to visualize relationships between features
plt.figure(figsize=(12,8))
sns.heatmap(merged_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
