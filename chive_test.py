######################################################################
#ChiVe README example
######################################################################
from pymagnitude import Magnitude

vectors = Magnitude("/home/eo/Desktop/fukushi/chive-1.0-mc5.magnitude")

vectors.query("あっという間に")

vectors.most_similar("流石に", topn=5)

vectors.similarity("あっという間に", "流石に")

######################################################################
#EMBED
######################################################################
import pandas as pd
import numpy as np
import pickle

input_file_path = '/home/eo/Desktop/fukushi/normalized_list.csv'
df = pd.read_csv(input_file_path, header=None)

input_file_path = '/home/eo/Desktop/fukushi/normalized_list.csv'

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

# Save adverb_array and embedding_array to NumPy binary files
np.save('/home/eo/Desktop/fukushi/adverb_array.npy', adverb_array)
np.save('/home/eo/Desktop/fukushi/embedding_array.npy', embedding_array)

# Load the numpy arrays
#adverb_array = np.load('/home/eo/Desktop/fukushi/adverb_array.npy')
#embedding_array = np.load('/home/eo/Desktop/fukushi/embedding_array.npy')


# Create a dictionary linking adverbs to embeddings
adverb_embedding_dict = dict(zip(adverb_array, embedding_array))

# Save the dictionary to a file using pickle
with open('/home/eo/Desktop/fukushi/adverb_embedding_dict.pkl', 'wb') as file:
    pickle.dump(adverb_embedding_dict, file)

# Load the dictionary from the pickle file
#with open('/home/eo/Desktop/fukushi/adverb_embedding_dict.pkl', 'rb') as file:
#    adverb_embedding_dict = pickle.load(file)

######################################################################
#K-means clustering
######################################################################

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

# Assuming you have loaded or created adverb_embedding_dict
# Convert the dictionary values (embeddings) to a NumPy array
embedding_matrix = np.array(list(adverb_embedding_dict.values()))

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(embedding_matrix)

######################################################################
#PCA 
######################################################################

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
embedding_2d_pca = pca.fit_transform(embedding_matrix)

# Specify a font that supports Japanese characters
japanese_font = FontProperties(fname='/home/eo/Desktop/fukushi/IPAexfont00401/ipaexm.ttf')  

# Plot the clusters in 2D with adverb labels
plt.figure(figsize=(10, 6))
for i in range(3):
    cluster_points = embedding_2d_pca[cluster_labels == i]
    cluster_adverbs = np.array(list(adverb_embedding_dict.keys()))[cluster_labels == i]
    
    for point, adverb in zip(cluster_points, cluster_adverbs):
        plt.scatter(point[0], point[1], label=f'Cluster {i}', marker='.')
        plt.text(point[0], point[1], adverb, fontsize=8, fontproperties=japanese_font)

plt.title('K-Means Clustering of Embedding Vectors with Adverb Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Create a dictionary linking adverbs to their cluster numbers
PCA_cluster_dict = dict(zip(adverb_embedding_dict.keys(), cluster_labels))

# Save the PCA_cluster_dict dictionary to a file using pickle
with open('/home/eo/Desktop/fukushi/PCA_cluster_dict.pkl', 'wb') as file:
    pickle.dump(PCA_cluster_dict, file)


######################################################################
#t-SNE
######################################################################

# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=42)
embedding_2d_tsne = tsne.fit_transform(embedding_matrix)

# Create a dictionary linking adverbs to their cluster numbers
tsne_cluster_dict = dict(zip(adverb_embedding_dict.keys(), cluster_labels))

# Specify a font that supports Japanese characters
japanese_font = FontProperties(fname='/home/eo/Desktop/fukushi/IPAexfont00401/ipaexm.ttf')  

# Plot the clusters in 2D with adverb labels
plt.figure(figsize=(10, 6))
for i in range(3):
    cluster_points = embedding_2d_tsne[cluster_labels == i]
    cluster_adverbs = np.array(list(adverb_embedding_dict.keys()))[cluster_labels == i]
    
    for point, adverb in zip(cluster_points, cluster_adverbs):
        plt.scatter(point[0], point[1], label=f'Cluster {i}', marker='.')
        plt.text(point[0], point[1], adverb, fontsize=8, fontproperties=japanese_font)

plt.title('K-Means Clustering of Embedding Vectors with Adverb Labels (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()


# Save the PCA_cluster_dict dictionary to a file using pickle
with open('/home/eo/Desktop/fukushi/tsne_cluster_dict.pkl', 'wb') as file:
    pickle.dump(PCA_cluster_dict, file)

######################################################################
#Confusion matrix comparison
######################################################################

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Extract the true labels and predicted labels from dictionaries
true_labels = list(PCA_cluster_dict.values())
predicted_labels = list(tsne_cluster_dict.values())

# Create a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.xlabel('t-SNE Predicted Cluster')
plt.ylabel('PCA True Cluster')
plt.title('Confusion Matrix: PCA vs t-SNE Clustering')
plt.show()

