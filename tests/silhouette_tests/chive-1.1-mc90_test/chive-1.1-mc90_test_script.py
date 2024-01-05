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
from sklearn.metrics import silhouette_score

######################################################################
#Load ChiVe model
######################################################################

vectors = Magnitude("./models/chive-1.1-mc90.magnitude")
test_directory = "./tests/silhouette_tests/chive-1.1-mc90_test/"
input_file_path = './adverbs/normalized_adverbs.csv'

######################################################################
#EMBED
######################################################################

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

# S

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
with open(test_directory+'adverb_embedding_dict.pkl', 'wb') as file:
    pickle.dump(adverb_embedding_dict, file)

######################################################################
#Start here if proceeding from before
######################################################################

# Load the dictionary from the pickle file
with open(test_directory+'adverb_embedding_dict.pkl', 'rb') as file:
    adverb_embedding_dict = pickle.load(file)

# Convert the dictionary values (embeddings) to a NumPy array
embedding_matrix = np.array(list(adverb_embedding_dict.values()))

######################################################################
# Silhouette Analysis
######################################################################

sil_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_matrix)
    silhouette_avg = silhouette_score(embedding_matrix, cluster_labels)
    sil_scores.append(silhouette_avg)

# Plot the Silhouette Score graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.title('Silhouette analysis for optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')

# Save the plots to the test_directory
plt.savefig(test_directory+'silhouette.png', dpi=1200)
