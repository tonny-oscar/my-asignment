from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select relevant features for segmentation
X = merged_df[['purchase_frequency', 'average_order_value', 'customer_lifetime_value']]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['customer_segment'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=merged_df['customer_segment'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Purchase Frequency')
plt.ylabel('Average Order Value')
plt.show()
