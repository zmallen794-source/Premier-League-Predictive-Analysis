import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, precision_score, recall_score

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
print(loadings.sort_values(by='PC1', key=abs, ascending=False).head(16))

kmeans= KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_pca_3d)

cluster_map = { 0: "Away Team Dominance", 1:"Home Team Dominance", 2: "Competitive"}
df_clean['ClusterLabel'] = df_clean['Cluster'].map(cluster_map)

summary_table = df_clean.groupby('ClusterLabel')[features].mean()
summary_table_transposed = summary_table.T

print("\n ---Cluster Summary Table (Feature Average) ---")
print(summary_table_transposed)

true_label_map = {'A': 'Away Team Dominance', 'H': 'Home Team Dominance', 'D':'Competitive'}
df_clean['TrueLabel'] = df_clean['FTR'].map(true_label_map)

precision = precision_score(df_clean['TrueLabel'], df_clean['ClusterLabel'], average='weighted')
recall = recall_score(df_clean['TrueLabel'], df_clean['ClusterLabel'], average='weighted')

print("\n --- Model Metrics---")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")

print("\n--- Classification Report---")
print(classification_report(df_clean['TrueLabel'], df_clean['ClusterLabel']))

#Visualization
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')

colors = ['#e74c3c', '#95a5a6', '#2ecc71']

for i, label in cluster_map.items():
	idx=df_clean['Cluster'] ==i
	ax.scatter(X_pca_3d[idx,0], X_pca_3d[idx,1], X_pca_3d[idx,2], c=colors[i], label=label, alpha=0.6, s=30)

explained = pca.explained_variance_ratio_
ax.set_xlabel(f'PC1: ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2: ({explained[1]*100:.1f}%)')
ax.set_zlabel(f'PC3: ({explained[2]*100:.1f}%)')
ax.set_title('Cluster Analysis of Football Matches')
ax.legend()

plt.show()

