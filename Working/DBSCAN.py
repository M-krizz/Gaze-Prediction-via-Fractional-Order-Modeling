import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Define the path where all user datasets are stored (workspace-relative)
# This resolves to the top-level `DataSet/` folder next to `Working/`.
data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataSet'))

# Initialize a list to store the preprocessed data for each user
preprocessed_data = []
user_names = []

# List all the CSV files for the 24 users
user_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]

# Loop over each file (representing each user's dataset)
for file_name in user_files:
    # Load the dataset
    file_path = os.path.join(data_directory, file_name)
    df = pd.read_csv(file_path)

    # Extract the gaze coordinates (x, y)
    gaze_data = df[['x', 'y']].values

    # Normalize the gaze data for each user (to compare them on a similar scale)
    gaze_data_normalized = StandardScaler().fit_transform(gaze_data)

    # Compute the mean and variance of the normalized gaze data to represent the user's gaze behavior
    user_features = np.mean(gaze_data_normalized, axis=0)  # Mean gaze position
    user_features = np.append(user_features, np.var(gaze_data_normalized, axis=0))  # Variance of gaze positions

    # Store the features and user name
    preprocessed_data.append(user_features)
    user_names.append(file_name.split('.')[0])  # Extract user ID from filename

# Convert preprocessed data into a DataFrame
features_df = pd.DataFrame(preprocessed_data, columns=['mean_x', 'mean_y', 'var_x', 'var_y'])
features_df['participant'] = user_names

# Calculate pairwise Euclidean distances between users (gaze behavior similarity)
similarity_matrix = euclidean_distances(features_df[['mean_x', 'mean_y', 'var_x', 'var_y']])

# Perform K-means clustering on the similarity matrix
kmeans = KMeans(n_clusters=4)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(similarity_matrix)

# Add the cluster labels to the DataFrame
features_df['cluster'] = clusters

# Print the players in each cluster
print("Players grouped by similar gaze patterns:")
for cluster_num in np.unique(clusters):
    players_in_cluster = features_df[features_df['cluster'] == cluster_num]['participant'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(players_in_cluster)}")

# Optionally, save the clustered data to a CSV
features_df.to_csv('clustered_players_based_on_gaze.csv', index=False)
