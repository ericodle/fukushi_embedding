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
test_directory = "/home/eo/Desktop/fukushi_embedding/tests/yamada_comparison/chive-1.2-mc5_test/"
manual_classification_path = '/home/eo/Desktop/fukushi_embedding/classifications/yamada_onehot.csv'

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

num_clusters = 3
# Custom display labels for the true_labels axis

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
# GMM analysis
######################################################################

# Create a GaussianMixture instance with 3 components (clusters)
gmm = GaussianMixture(n_components=num_clusters, random_state=42)

# Fit the model to the data and predict the Gaussian component responsibilities
gmm.fit(embedding_matrix)
gmm_cluster_probs = gmm.predict_proba(embedding_matrix)
gmm_cluster_labels = gmm.predict(embedding_matrix)

######################################################################
# Label permutations
######################################################################

# Generate all permutations of cluster labels
gmm_label_permutations = list(permutations(range(num_clusters)))

# Create variables for each permutation
gmm_cluster_labels_permutations = {}

for i, perm in enumerate(gmm_label_permutations):
    # Create a new variable for each permutation, e.g., cluster_labels_0, cluster_labels_1, ...
    var_name = f'cluster_labels_{i}'
    
    # Swap the cluster labels based on the permutation
    gmm_cluster_labels_permutations[var_name] = [perm[label] for label in gmm_cluster_labels]

#print(cluster_labels_permutations)

######################################################################
# gmm accuracy calculation
######################################################################

# Dictionary to store accuracies for each permutation
gmm_accuracies = {}

# Iterate over each permutation
for var_name, perm_labels in gmm_cluster_labels_permutations.items():
    # Calculate accuracy for the current permutation
    accuracy = accuracy_score(true_labels, perm_labels)
    
    # Store the accuracy in the dictionary
    gmm_accuracies[var_name] = accuracy



######################################################################
# gmm Result
######################################################################

# Assuming accuracies dictionary is already populated
# Custom display labels for the true_labels axis
true_labels_display_labels = ["Status", "Degree", "Declarative"]

# Find the key with the highest accuracy
gmm_best_result_key = max(gmm_accuracies, key=gmm_accuracies.get)

# Get the cluster labels for the best result
gmm_best_result_labels = gmm_cluster_labels_permutations[gmm_best_result_key]

# Print the best result accuracy
gmm_best_result_accuracy = gmm_accuracies[gmm_best_result_key]
#print(f"Best Result (Cluster Labels {best_result_key}): {best_result_accuracy}")

# Truncate accuracy to the last 3 digits after the decimal point
gmm_truncated_accuracy = "{:.3f}".format(gmm_best_result_accuracy)

# Generate confusion matrix plot for the best result with custom display labels
gmm_conf_matrix = confusion_matrix(true_labels, gmm_best_result_labels)
disp = ConfusionMatrixDisplay(gmm_conf_matrix, display_labels=true_labels_display_labels)
disp.plot(cmap='Blues', values_format='d', xticks_rotation='horizontal', ax=plt.gca())
plt.title(f'Confusion Matrix | Acc: {gmm_truncated_accuracy}')
plt.tight_layout()

# Save the plots to the test_directory
plt.savefig(test_directory+'gmm_accuracy.png')

