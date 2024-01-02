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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import permutations


######################################################################
# Load
######################################################################
vectors = Magnitude("/home/eo/Desktop/fukushi_embedding/models/chive-1.2-mc5.magnitude")
test_directory = "/home/eo/Desktop/fukushi_embedding/tests/noda_comparisons/chive-1.2-mc5_test/"
manual_classification_path = '/home/eo/Desktop/fukushi_embedding/classifications/noda_onehot.csv'

manual_df = pd.read_csv(manual_classification_path, header=None, names=None)

adverb_list = manual_df.iloc[:, 0].tolist()
#print("adverb_list")
#print(len(adverb_list))
#print(adverb_list)

# Process manual_class_columns based on the specified conditions
manual_class_columns = manual_df.iloc[:, 1:].apply(lambda x: x.idxmax(), axis=1)
# Fill NaN values with 0
manual_class_columns = manual_class_columns.fillna(0).astype(int)

#because zero-indexed labeling
true_labels = manual_class_columns - 1
#print("true_labels")
#print(len(true_labels))
#print(true_labels)

num_clusters = 5

######################################################################
# Embed
######################################################################

def chive_embed(adverb):
    adverb_str = str(adverb)  # Convert adverb to string
    try:
        embedded_adverb = vectors.query(adverb_str)
        return np.array(embedded_adverb)  # Convert to NumPy array
    except KeyError:
        return None

embedding_matrix = [chive_embed(adverb) for adverb in adverb_list]
#print("embedding_matrix")
#print(len(embedding_matrix))
#print(embedding_matrix)

######################################################################
#k-means cluster
######################################################################

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_cluster_labels = kmeans.fit_predict(embedding_matrix)


#print("cluster_labels")
#print(len(cluster_labels))
#print(cluster_labels)

######################################################################
# Label permutations
######################################################################

# Generate all permutations of cluster labels
label_permutations = list(permutations(range(num_clusters)))

# Create variables for each permutation
cluster_labels_permutations = {}

for i, perm in enumerate(label_permutations):
    # Create a new variable for each permutation, e.g., cluster_labels_0, cluster_labels_1, ...
    var_name = f'cluster_labels_{i}'
    
    # Swap the cluster labels based on the permutation
    cluster_labels_permutations[var_name] = [perm[label] for label in kmeans_cluster_labels]

#print(cluster_labels_permutations)

######################################################################
# Kmeans accuracy calculation
######################################################################

# Dictionary to store accuracies for each permutation
kmeans_accuracies = {}

# Iterate over each permutation
for var_name, perm_labels in cluster_labels_permutations.items():
    # Calculate accuracy for the current permutation
    accuracy = accuracy_score(true_labels, perm_labels)
    
    # Store the accuracy in the dictionary
    kmeans_accuracies[var_name] = accuracy


######################################################################
# Kmeans Result
######################################################################

# Assuming accuracies dictionary is already populated
# Custom display labels for the true_labels axis
true_labels_display_labels = ["Mood", "Tense", "Aspect", "Voice", "Objects"]

# Find the key with the highest accuracy
best_result_key = max(kmeans_accuracies, key=kmeans_accuracies.get)

# Get the cluster labels for the best result
best_result_labels = cluster_labels_permutations[best_result_key]

# Print the best result accuracy
best_result_accuracy = kmeans_accuracies[best_result_key]
#print(f"Best Result (Cluster Labels {best_result_key}): {best_result_accuracy}")

# Truncate accuracy to the last 3 digits after the decimal point
kmeans_truncated_accuracy = "{:.3f}".format(best_result_accuracy)

# Generate confusion matrix plot for the best result with custom display labels
kmeans_conf_matrix = confusion_matrix(true_labels, best_result_labels)
disp = ConfusionMatrixDisplay(kmeans_conf_matrix, display_labels=true_labels_display_labels)
disp.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal', ax=plt.gca())
plt.title(f'Confusion Matrix | Acc: {kmeans_truncated_accuracy}')
plt.tight_layout()

# Save the plots to the test_directory
plt.savefig(test_directory+'kmeans_accuracy.png')


