import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load your dataset ======
# Replace with your actual dataset path
df = pd.read_csv("DataSet/P01_PLAY.csv")

# Ensure timestamp is sorted
df = df.sort_values("timestamp").reset_index(drop=True)

# ====== Plot trajectory ======
def plot_trajectory(data, participant_id=None):
    plt.figure(figsize=(7, 6))
    plt.plot(data["x"], data["y"], color="blue", linewidth=1, alpha=0.7)
    plt.scatter(data["x"], data["y"], s=5, c=data["timestamp"], cmap="viridis")
    plt.colorbar(label="Timestamp")
    plt.title(f"Eye Movement Trajectory {participant_id}")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.gca().invert_yaxis()  # optional: match screen coordinates
    plt.show()

# Example: plot trajectory of one participant
plot_trajectory(df, participant_id=df["participant"].iloc[0])

# ====== Heatmap of gaze density ======
def plot_heatmap(data, participant_id=None, bins=50):
    plt.figure(figsize=(7, 6))
    sns.kdeplot(
        x=data["x"], y=data["y"],
        fill=True, cmap="mako", bw_adjust=0.5, levels=50, thresh=0.05
    )
    plt.title(f"Gaze Heatmap {participant_id}")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.gca().invert_yaxis()
    plt.show()

# Example: plot heatmap of the same participant
plot_heatmap(df, participant_id=df["participant"].iloc[0])
