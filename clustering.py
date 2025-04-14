import pandas as pd
from kmodes.kmodes import KModes  
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv(os.path.join('data', 'preprocessed_data.csv'))

# Select relevant features for clustering
X = df[['ProductType', 'Colour']].dropna()
X_encoded = X.copy()
X_encoded['ProductType'] = X_encoded['ProductType'].astype(str)
X_encoded['Colour'] = X_encoded['Colour'].astype(str)

# KModes clustering (changed from KPrototypes to KModes for categorical data)
kmodes = KModes(n_clusters=4, init='Cao', verbose=0)
clusters = kmodes.fit_predict(X_encoded)
X_encoded['Cluster'] = clusters

# Merge cluster label back to main dataframe
df = df.merge(X_encoded, how='left', on=['ProductType', 'Colour'])

# Feature Engineering on InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Hour'] = df['InvoiceDate'].dt.hour
df['Weekday'] = df['InvoiceDate'].dt.weekday

# Plot clustered data by hour and weekday
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Hour', hue='Cluster', palette='Set1')
plt.title('Distribution of Clusters by Hour')
plt.tight_layout()
plt.savefig(os.path.join('metrics', 'cluster_by_hour.png'))
plt.show()

# Save clustered data
df.to_csv(os.path.join('data', 'clustered_data.csv'), index=False)
print("[✓] Clustered dataset saved.")
