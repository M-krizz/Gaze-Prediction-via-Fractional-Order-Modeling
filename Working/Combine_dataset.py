import pandas as pd
import glob
import os

# Path where all CSVs are stored
path = "DataSet"

# Get all CSV file paths
all_files = glob.glob(os.path.join(path, "*.csv"))

# Read and combine
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save as single CSV
combined_df.to_csv("combined_dataset.csv", index=False)

print("Combined file saved as combined_dataset.csv")
