import matplotlib.pyplot as plt
import seaborn as sns

# Initial data description
print(merged_df.describe())

# Visualize distributions
plt.figure(figsize=(10,6))
sns.histplot(merged_df['sales'], kde=True)
plt.title('Distribution of Sales')
plt.show()

# Visualize correlations
plt.figure(figsize=(10,6))
sns.heatmap(merged_df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()
