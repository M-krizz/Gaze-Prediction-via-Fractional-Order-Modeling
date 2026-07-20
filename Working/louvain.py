import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import networkx as nx
import community as community_louvain

# Truncated, preprocessed participant recordings.
data_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'trucate_files', 'raw')
)

# Initialize a dictionary to store players' data by clusters
clustered_data = {}

# List all the CSV files for the 24 users
user_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]

# Loop over each file (representing each user's dataset)
for file_name in user_files:
    # Load the dataset
    file_path = os.path.join(data_directory, file_name)
    df = pd.read_csv(file_path)

    # Extract user ID from file name
    participant_id = file_name.split('.')[0]

    # Normalize the gaze data (x, y coordinates)
    gaze_data = df[['x', 'y']].values
    gaze_data_normalized = StandardScaler().fit_transform(gaze_data)

    # Create a graph where nodes are the participants, and edges represent gaze similarity
    G = nx.Graph()

    # Adding nodes for each player
    G.add_node(participant_id, features=gaze_data_normalized)

    # Add edges based on similarity (Euclidean distance between players' gaze features)
    for other_file_name in user_files:
        if file_name != other_file_name:  # Skip comparing the same player to themselves
            other_file_path = os.path.join(data_directory, other_file_name)
            other_df = pd.read_csv(other_file_path)
            other_participant_id = other_file_name.split('.')[0]

            # Compute similarity (e.g., Euclidean distance between players' gaze features)
            other_gaze_data = other_df[['x', 'y']].values
            other_gaze_data_normalized = StandardScaler().fit_transform(other_gaze_data)

            # Compute Euclidean distance between the current player and the other player
            distance = np.linalg.norm(np.mean(gaze_data_normalized, axis=0) - np.mean(other_gaze_data_normalized, axis=0))
            
            # Create an edge with weight as similarity (inverse of distance)
            G.add_edge(participant_id, other_participant_id, weight=1 / (1 + distance))  # Higher weight for smaller distance

# Apply Louvain method for community detection
partition = community_louvain.best_partition(G, weight='weight')

# Print the players grouped by similar gaze patterns (Louvain method)
print("Players grouped by similar gaze patterns (Louvain communities):")
for community_id in set(partition.values()):
    players_in_community = [player for player, community in partition.items() if community == community_id]
    print(f"Community {community_id}: {', '.join(players_in_community)}")

# Initialize a dictionary to store players' data by clusters (from Louvain method)
for file_name in user_files:
    # Load the dataset again
    file_path = os.path.join(data_directory, file_name)
    df = pd.read_csv(file_path)
    
    # Extract user ID from file name
    participant_id = file_name.split('.')[0]
    
    # Get the cluster label for this user from the partition (Louvain method)
    cluster_label = partition.get(participant_id)

    # Add the participant data to the corresponding cluster
    if cluster_label not in clustered_data:
        clustered_data[cluster_label] = []
    
    clustered_data[cluster_label].append(df)

# Loop over each cluster and compute the mean of x, y, and timestamps for each row
for cluster, players_data in clustered_data.items():
    print(f"Processing Cluster {cluster}")

    # Find the dataset with the minimum number of rows (base dataset)
    min_rows_player = min(players_data, key=lambda x: len(x))  # Get player with the smallest dataset
    base_player_id = min_rows_player['participant'].iloc[0]

    # Initialize a dataframe to store the means for each row
    mean_df = min_rows_player[['x', 'y', 'timestamp']].copy()  # Start with the first player's data

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

    # Add the participant ID and cluster label to the dataframe (using the base player's info)
    mean_df['participant'] = base_player_id
    mean_df['cluster'] = cluster

    # Save the output for the cluster (mean of all rows for the cluster)
    mean_df.to_csv(f'cluster_{cluster}_mean.csv', index=False)

    print(f"Cluster {cluster} mean dataset saved as 'cluster_{cluster}_mean.csv'")

# Optionally, save the final partitioned data (player clusters) based on Louvain method
partition_df = pd.DataFrame(list(partition.items()), columns=['participant', 'community'])
partition_df.to_csv('louvain_community_partition.csv', index=False)
