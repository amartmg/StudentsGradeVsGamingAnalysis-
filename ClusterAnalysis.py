import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

os.makedirs("Images", exist_ok=True)

# Load the cleaned data
df = pd.read_csv("gameandgrade_cleaned.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ── 1. SELECT FEATURES ─────────────────────────────────────────────────────
# We use gaming habits + parental education + Grade
# Playing Often is excluded (correlation with Grade = 0.009, not useful)
features = ["Playing Hours", "Playing Games", "Playing Years",
            "Mother Education", "Father Education", "Grade"]

X = df[features]

# ── 2. NORMALIZE THE DATA ──────────────────────────────────────────────────
# KMeans uses distance to group students
# Without normalization, Grade (33-100) would dominate over Playing Hours (0-5)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. ELBOW METHOD — Find the best number of clusters ────────────────────
# We test K=1 to K=10 and look for where the curve flattens
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker="o", color="blue")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method: Finding Best K")
plt.savefig("Images/Elbow_Method.png")
plt.show()

# ── 4. APPLY KMEANS WITH K=3 ───────────────────────────────────────────────
# K=3 chosen from elbow curve — curve flattens after K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ── 5. CLUSTER SUMMARY ─────────────────────────────────────────────────────
print("\nNumber of students per cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nAverage values per cluster:")
print(df.groupby("Cluster")[features].mean().round(2))

# Label clusters based on average Grade
# Cluster 0: Moderate Gamers  (avg Grade 75.70, avg Hours 1.70)
# Cluster 1: Heavy Gamers     (avg Grade 76.07, avg Hours 2.72)
# Cluster 2: Non-Gamers       (avg Grade 81.51, avg Hours 0.00)

# ── 6. VISUALIZE THE CLUSTERS ──────────────────────────────────────────────
colors = ["orange", "red", "green"]
labels = ["Cluster 0: Moderate Gamers",
          "Cluster 1: Heavy Gamers",
          "Cluster 2: Non-Gamers"]

plt.figure(figsize=(8, 5))
for i in range(3):
    cluster_data = df[df["Cluster"] == i]
    plt.scatter(cluster_data["Playing Hours"], cluster_data["Grade"],
                color=colors[i], alpha=0.5, s=20, label=labels[i])

plt.xlabel("Playing Hours")
plt.ylabel("Grade")
plt.title("Cluster Analysis: Student Groups")
plt.legend()
plt.tight_layout()
plt.savefig("Images/Cluster_Analysis.png")
plt.show()

print("\nCluster Analysis complete!")
print("Charts saved: Elbow_Method.png, Cluster_Analysis.png")