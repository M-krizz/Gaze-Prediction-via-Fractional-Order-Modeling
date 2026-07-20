import pandas as pd
import os
import numpy as np
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

# Perform Spectral Clustering on the similarity matrix
spectral = SpectralClustering(n_clusters=4, affinity='precomputed', n_init=100)
clusters = spectral.fit_predict(similarity_matrix)

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

# Define a simple GCN model for graph feature extraction
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Function to create graph data for GCN
def create_graph_data(nodes, edge_index):
    x = torch.tensor(np.asarray(nodes), dtype=torch.float)
    edge_index = torch.tensor(np.asarray(edge_index).T, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# Example graph construction (assuming gaze data with 2D features as nodes)
for cluster, players_data in clustered_data.items():
    print(f"Processing Cluster {cluster}")
    
    # Initialize an empty graph
    nodes = []  # Will contain node features (mean_x, mean_y, var_x, var_y)
    edge_index = []  # Define edges between players (this can be based on some similarity)
    
    # Loop through each player's dataset and add node features
    for player_df in players_data:
        player_features = player_df[['x', 'y']].mean(axis=0).values  # Example: mean of x, y as features
        nodes.append(player_features)
    
    # For simplicity, create a fully connected graph (edges between all nodes)
    num_nodes = len(nodes)
    edge_index = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    # Create graph data object for GCN
    graph_data = create_graph_data(nodes, edge_index)

    # Apply GCN to get node embeddings
    model = GCN(num_features=2, num_classes=4)  # Adjust the number of features/classes
    out = model(graph_data)
    print(f"Graph embeddings for cluster {cluster}: {out}")
    
    # Here you would integrate GCN-based clustering (optional)

    # Save the final output for each cluster
    final_df = pd.DataFrame(nodes, columns=['mean_x', 'mean_y'])
    final_df['cluster'] = cluster
    final_df.to_csv(f'cluster_{cluster}_final.csv', index=False)

    print(f"Cluster {cluster} final dataset saved as 'cluster_{cluster}_final.csv'")
