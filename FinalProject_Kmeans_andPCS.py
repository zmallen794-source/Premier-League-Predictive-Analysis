import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

#Load and Prepare Data
df= pd.read_csv('results.csv', encoding='latin1')
features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
df_clean = df.dropna(subset=features).copy()

#Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

pca = PCA(n_components =3)
X_pca_3d = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=features)
loadings['influence'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2 + loadings['PC3']**2)
top_3_features = loadings['influence'].sort_values(ascending=False).head(3)

print("Top 3 Features Identified by PCA:")
print(top_3_features)

kmeans= KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_pca_3d)

cluster_map = { 0: "Away Team Dominance", 1:"Competitve", 2: "Home Team Dominance"}
df_clean['ClusterLabel'] = df_clean['Cluster'].map(cluster_map)

#Visualization
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

colors = ['#e74c3c', '#95a5a6', '#2ecc71']

for i, label in cluster_map.items():
	idx=df_clean['Cluster'] ==i
	ax.scatter(X_pca_3d[idx,0], X_pca_3d[idx,1], X_pca_3d[idx,2], c=colors[i], label=label, alpha=0.6, s=30)


ax.set_xlabel('PC1: Shot Volume')
ax.set_ylabel('PC2: Goal Scoring')
ax.set_zlabel('PC3: Discipline & Away Pressure')
ax.set_title('Cluster Analysis of Football Matches')
ax.legend()

plt.show()

