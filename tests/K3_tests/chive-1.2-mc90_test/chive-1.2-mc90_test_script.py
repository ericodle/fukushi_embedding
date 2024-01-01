######################################################################
#Imports
######################################################################
from pymagnitude import Magnitude
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
import seaborn as sns


######################################################################
#Load ChiVe model
######################################################################

vectors = Magnitude("/home/eo/Desktop/fukushi_embedding/models/chive-1.2-mc90.magnitude")

test_directory = "/home/eo/Desktop/fukushi_embedding/tests/chive-1.2-mc90_test/"

######################################################################
#EMBED
######################################################################


input_file_path = '/home/eo/Desktop/fukushi_embedding/adverbs/normalized_adverbs.csv'
df = pd.read_csv(input_file_path, header=None)

def chive_embed(adverb):
    adverb_str = str(adverb)  # Convert adverb to string
    try:
        embedded_adverb = vectors.query(adverb_str)
        return np.array(embedded_adverb)  # Convert to NumPy array
    except KeyError:
        return None

adverb_list = df[0].tolist()

embedding_list = [chive_embed(adverb) for adverb in adverb_list]

adverb_array = np.array(adverb_list)
embedding_array = np.array(embedding_list)

######################################################################
#K-means clusteringave adverb_array and embedding_array to NumPy binary files
np.save(test_directory+'adverb_array.npy', adverb_array)
np.save(test_directory+'embedding_array.npy', embedding_array)

# Load the numpy arrays
#adverb_array = np.load('/home/eo/Desktop/fukushi/adverb_array.npy')
#embedding_array = np.load('/home/eo/Desktop/fukushi/embedding_array.npy')


# Create a dictionary linking adverbs to embeddings
adverb_embedding_dict = dict(zip(adverb_array, embedding_array))

# Save the dictionary to a file using pickle
with open(test_directory+'/adverb_embedding_dict.pkl', 'wb') as file:
    pickle.dump(adverb_embedding_dict, file)

######################################################################
#Start here if proceeding mid-way
######################################################################

# Load the dictionary from the pickle file
#with open('/home/eo/Desktop/fukushi_embedding/tests/20231221/adverb_embedding_dict.pkl', 'rb') as file:
#    adverb_embedding_dict = pickle.load(file)

# Convert the dictionary values (embeddings) to a NumPy array
embedding_matrix = np.array(list(adverb_embedding_dict.values()))

######################################################################
#k-means
######################################################################

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels, palette='viridis')
plt.title('K-Means Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels, palette='viridis')
plt.title('K-Means Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'kmeans.png')

######################################################################
#gmm
######################################################################

# Create a GaussianMixture instance with 3 components (clusters)
gmm = GaussianMixture(n_components=3, random_state=42)

# Fit the model to the data and predict the Gaussian component responsibilities
gmm.fit(embedding_matrix)
gmm_cluster_probs = gmm.predict_proba(embedding_matrix)
gmm_cluster_labels = gmm.predict(embedding_matrix)


# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=gmm_cluster_labels, palette='viridis')
plt.title('GMM Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=gmm_cluster_labels, palette='viridis')
plt.title('GMM Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'gmm.png')

####################################################################
#Hierarchical clustering
#####################################################################


# Perform Hierarchical Clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=3)
cluster_labels_hierarchical = hierarchical_clustering.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_hierarchical, palette='viridis')
plt.title('Hierarchical Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_hierarchical, palette='viridis')
plt.title('Hierarchical Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'hierarchical.png')

####################################################################
#DBSCAN
#####################################################################

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels_dbscan = dbscan.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_dbscan, palette='viridis')
plt.title('DBSCAN Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_dbscan, palette='viridis')
plt.title('DBSCAN Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'dbscan.png')

####################################################################
#affinityclustering
#####################################################################

# Perform Affinity Propagation Clustering
affinity_propagation = AffinityPropagation()
cluster_labels_affinity = affinity_propagation.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_affinity, palette='viridis')
plt.title('Affinity Propagation Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_affinity, palette='viridis')
plt.title('Affinity Propagation Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'affinity.png')

####################################################################
#mean shift
#####################################################################
# Perform Mean Shift Clustering
mean_shift = MeanShift()
cluster_labels_mean_shift = mean_shift.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_mean_shift, palette='viridis')
plt.title('Mean Shift Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_mean_shift, palette='viridis')
plt.title('Mean Shift Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'meanshift.png')

####################################################################
#spectralclustering
####################################################################

# Perform Spectral Clustering
spectral_clustering = SpectralClustering(n_clusters=3, random_state=42)
cluster_labels_spectral = spectral_clustering.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_spectral, palette='viridis')
plt.title('Spectral Clustering with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_spectral, palette='viridis')
plt.title('Spectral Clustering with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'spectral.png')

####################################################################
#selforganizingmap(SOM) via minibatch kmeans
####################################################################


# Perform MiniBatchKMeans (as an approximation to SOM)
som = MiniBatchKMeans(n_clusters=3, random_state=42)
cluster_labels_som = som.fit_predict(embedding_matrix)

# Apply PCA for visualization
pca = PCA(n_components=2)
embedding_pca = pca.fit_transform(embedding_matrix)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_matrix)

# Plotting with PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=embedding_pca[:, 0], y=embedding_pca[:, 1], hue=cluster_labels_som, palette='viridis')
plt.title('Self-Organizing Maps (SOM) with PCA')

# Plotting with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=cluster_labels_som, palette='viridis')
plt.title('Self-Organizing Maps (SOM) with t-SNE')

# Save the plots to the test_directory
plt.savefig(test_directory+'som_minikmeans.png')

