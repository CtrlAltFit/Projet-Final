from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

# Fonction pour la réduction de dimension
def dim_red(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

# Fonction pour effectuer le clustering avec k-means
def clust(X, k):
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(X)

# Import des données
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]

# Modèle d'embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Pipeline avec ACP et k-means
pipeline = make_pipeline(PCA(n_components=20), KMeans(n_clusters=len(set(labels))))

# Ajustement du modèle
pipeline.fit(embeddings)

# Prédiction des clusters
pred = pipeline.predict(embeddings)

# Évaluation des résultats de clustering
nmi_score = normalized_mutual_info_score(pred, labels)
ari_score = adjusted_rand_score(pred, labels)

print(f'NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame for visualization
import pandas as pd
df = pd.DataFrame({'embedding_1': embeddings[:, 0], 'embedding_2': embeddings[:, 1], 'cluster': pred})

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='embedding_1', y='embedding_2', hue='cluster', data=df, palette='viridis')
plt.title('Scatter Plot of Embeddings with Clusters')
plt.show()
# Assuming you have already defined and fitted the PCA model
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Scatter plot for PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pred, cmap='viridis')
plt.title('PCA: Scatter Plot of Reduced Dimensions with Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Create a DataFrame for visualization
df_purity = pd.DataFrame({'true_label': labels, 'cluster': pred})

# Count occurrences of each true label within each cluster
purity_counts = df_purity.groupby(['cluster', 'true_label']).size().unstack().fillna(0)

# Normalize the counts to get the distribution
purity_distribution = purity_counts.div(purity_counts.sum(axis=1), axis=0)

# Bar chart
purity_distribution.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
plt.title('Cluster Purity: True Label Distribution within Predicted Clusters')
plt.xlabel('Predicted Cluster')
plt.ylabel('Distribution')
plt.show()

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
!pip install sentence_transformers
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

# Fonction pour effectuer la réduction de dimension avec t-SNE
def dim_red(X, n_components):
    tsne = TSNE(n_components=n_components)
    return tsne.fit_transform(X)

# Fonction pour effectuer le clustering avec k-means
def clust(X, k):
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(X)

# Wrapper pour utiliser la fonction dans le pipeline
tsne_transformer = FunctionTransformer(dim_red, kw_args={'n_components': 3})

# Import des données
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]

# Modèle d'embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Pipeline avec t-SNE et k-means
pipeline = make_pipeline(tsne_transformer, KMeans(n_clusters=len(set(labels))))

# Ajustement du modèle
pipeline.fit(embeddings)

# Prédiction des clusters
pred = pipeline.predict(embeddings)

# Évaluation des résultats de clustering
nmi_score = normalized_mutual_info_score(pred, labels)
ari_score = adjusted_rand_score(pred, labels)

print(f'NMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=pred, cmap='viridis', alpha=0.7)
ax.set_title('t-SNE Visualization of Clusters (3D)')
plt.show()

# Visualize the 2D clusters using t-SNE
tsne_2d = TSNE(n_components=2)
embeddings_2d = tsne_2d.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pred, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Clusters (2D)')
plt.show()

pred = pipeline.predict(embeddings)
import numpy as np
# Bar chart for cluster distribution
cluster_counts = np.bincount(pred)
plt.figure(figsize=(8, 6))
plt.bar(range(len(cluster_counts)), cluster_counts, color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Number of Data Points in Each Cluster')
plt.show()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]

# Model embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Function for dimensionality reduction and clustering
def perform_dimensionality_reduction_and_clustering(X, method, k):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError("Invalid method. Use 'PCA' or 'tsne'.")

    # Perform dimensionality reduction
    red_emb = reducer.fit_transform(X)

    # Perform clustering
    kmeans = KMeans(n_clusters=k)
    pred = kmeans.fit_predict(red_emb)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}')

    # Visualize the results
    visualize_results(red_emb, pred, method)

# Function to visualize the results
def visualize_results(embedding, pred, method):
    # Create a DataFrame for visualization
    df = pd.DataFrame({'embedding_1': embedding[:, 0], 'embedding_2': embedding[:, 1], 'cluster': pred})

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='embedding_1', y='embedding_2', hue='cluster', data=df, palette='viridis')
    plt.title(f'Scatter Plot of Embeddings with Clusters ({method})')
    plt.show()

# Perform dimensionality reduction and clustering for PCA
perform_dimensionality_reduction_and_clustering(embeddings, method='PCA', k=len(set(labels)))

# Perform dimensionality reduction and clustering for t-SNE
perform_dimensionality_reduction_and_clustering(embeddings, method='tsne', k=len(set(labels)))


