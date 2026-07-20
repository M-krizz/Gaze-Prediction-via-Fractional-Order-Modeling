import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances

# Truncated, preprocessed participant recordings.
data_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'trucate_files', 'raw')
)

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

from sklearn.metrics import silhouette_score

best_k = None
best_score = -1
best_clusters = None

print("Evaluating different numbers of clusters:\n")

# Number of clusters from 2 to 10
# (Silhouette Score cannot be computed for k=1)
for k in range(2, 11):
    spectral = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        n_init=100,
        random_state=42
    )

    clusters = spectral.fit_predict(similarity_matrix)

    score = silhouette_score(
        features_df[['mean_x', 'mean_y', 'var_x', 'var_y']],
        clusters
    )

    print(f"Clusters = {k:2d} | Silhouette Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k
        best_clusters = clusters

print("\n--------------------------------")
print(f"Best Number of Clusters : {best_k}")
print(f"Best Silhouette Score   : {best_score:.4f}")
print("--------------------------------\n")

# Use the best clustering for the rest of the code
clusters = best_clusters

# Add the cluster labels to the DataFrame
features_df['cluster'] = clusters

# Print the players in each cluster
print("Players grouped by similar gaze patterns:")
for cluster_num in np.unique(clusters):
    players_in_cluster = features_df[features_df['cluster'] == cluster_num]['participant'].tolist()
    print(f"Cluster {cluster_num}: {', '.join(players_in_cluster)}")

# Initialize a dictionary to store players' data by clusters
clustered_data = {}

# Loop over each file again to add data to each cluster
for file_name in user_files:
    # Load the dataset again
    file_path = os.path.join(data_directory, file_name)
    df = pd.read_csv(file_path)
    
    # Extract user ID from file name
    participant_id = file_name.split('.')[0]
    
    # Get the cluster label for this user
    cluster_label = features_df[features_df['participant'] == participant_id]['cluster'].iloc[0]

    # Add the participant data to the corresponding cluster
    if cluster_label not in clustered_data:
        clustered_data[cluster_label] = []
    
    clustered_data[cluster_label].append(df)

# Loop over each cluster and compute the mean of x, y, and timestamps for each row
for cluster, players_data in clustered_data.items():
    print(f"Processing Cluster {cluster}")

    # Initialize a dataframe to store the means for each row
    mean_df = players_data[0][['x', 'y', 'timestamp']].copy()  # Start with the first player's data

    # Loop over each player dataset in the cluster
    for player_df in players_data:
        # Calculate the mean of x, y, and timestamp for each row across all players
        mean_df['x'] += player_df['x']
        mean_df['y'] += player_df['y']
        mean_df['timestamp'] += player_df['timestamp']

    # Average the features across all players in the cluster
    num_players = len(players_data)
    mean_df['x'] /= num_players
    mean_df['y'] /= num_players
    mean_df['timestamp'] /= num_players

    # Add the participant ID and cluster label to the dataframe (using the first player's info)
    mean_df['participant'] = players_data[0]['participant'].iloc[0]  # Base participant ID (any player from the cluster)
    mean_df['cluster'] = cluster

    # Save the output for the cluster (mean of all rows for the cluster)
    mean_df.to_csv(f'cluster_{cluster}_mean.csv', index=False)

    print(f"Cluster {cluster} mean dataset saved as 'cluster_{cluster}_mean.csv'")

# Optionally, save the final clustered data (player clusters)
features_df.to_csv('clustered_players_based_on_gaze.csv', index=False)
