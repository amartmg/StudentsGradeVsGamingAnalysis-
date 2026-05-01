import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("Images", exist_ok=True)

# Load the cleaned data
df = pd.read_csv("gameandgrade_cleaned.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ── 1. ATTRIBUTE SUBSET SELECTION ──────────────────────────────────────────
# Check how much each column is related to Grade
# Values close to 1 or -1 = strong relationship
# Values close to 0 = weak relationship (not useful)
correlation = df.corr(numeric_only=True)["Grade"].sort_values(ascending=False)
print("\nCorrelation with Grade:")
print(correlation)

# Remove Playing Often — correlation = 0.009 (basically useless)
df_reduced = df.drop(columns=["Playing Often"])
print("\nColumns after reduction:", df_reduced.shape)

# ── 2. LINEAR REGRESSION ───────────────────────────────────────────────────
# Predict Grade using Mother Education (strongest single predictor)
X = df[["Mother Education"]]
y = df["Grade"]

model = LinearRegression()
model.fit(X, y)

print("\nLinear Regression Results:")
print(f"  Coefficient: {model.coef_[0]:.4f}")
print(f"  Intercept:   {model.intercept_:.4f}")
print(f"  Formula: Grade = {model.intercept_:.2f} + ({model.coef_[0]:.2f} x Mother Education)")

# Plot linear regression
plt.scatter(X, y, color="blue", alpha=0.3, label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Mother Education")
plt.ylabel("Grade")
plt.title("Linear Regression: Mother Education vs Grade")
plt.legend()
plt.savefig("Images/Linear_Regression.png")
plt.show()

# ── 3. MULTIPLE REGRESSION ─────────────────────────────────────────────────
# Use all columns together to predict Grade
X_multi = df_reduced.drop(columns=["Grade"])
y = df["Grade"]

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

print("\nMultiple Regression Results:")
print(f"  R² Score: {model_multi.score(X_multi, y):.4f}")
print("  Coefficients:")
for col, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"    {col}: {coef:.4f}")

# ── 4. HISTOGRAM ANALYSIS ──────────────────────────────────────────────────
# Shows the distribution of grades across all students
plt.hist(df["Grade"], bins=10, color="blue", edgecolor="white")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.title("Histogram: Grade Distribution")
plt.savefig("Images/Histogram.png")
plt.show()

# ── 5. CLUSTERING (K-Means) ────────────────────────────────────────────────
# Group students into 3 clusters based on Playing Hours and Grade
X_cluster = df[["Playing Hours", "Grade"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster)

print("\nClustering Results:")
print(df["Cluster"].value_counts())
print("\nAverage Grade per Cluster:")
print(df.groupby("Cluster")["Grade"].mean().round(2))

# Plot clusters
colors = ["orange", "green", "red"]
for i in range(3):
    cluster_data = df[df["Cluster"] == i]
    plt.scatter(cluster_data["Playing Hours"], cluster_data["Grade"],
                color=colors[i], alpha=0.5, label=f"Cluster {i}")
plt.xlabel("Playing Hours")
plt.ylabel("Grade")
plt.title("Clustering: Students Grouped by Playing Hours & Grade")
plt.legend()
plt.savefig("Images/Cluster.png")
plt.show()

# ── 6. SAMPLING ────────────────────────────────────────────────────────────
# Take 100 random students to represent the full dataset
sample = df.sample(n=100, random_state=42)
print("\nSampling Results:")
print(f"  Full dataset average Grade: {df['Grade'].mean():.2f}")
print(f"  Sample average Grade:       {sample['Grade'].mean():.2f}")

# ── 7. PCA ─────────────────────────────────────────────────────────────────
# Reduce 9 columns into fewer components while keeping most information
X_pca = df.drop(columns=["Grade", "Cluster"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA()
pca.fit(X_scaled)

print("\nPCA Results:")
cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var * 100
    print(f"  Component {i+1}: {var*100:.2f}% (cumulative: {cumulative:.2f}%)")

# Plot PCA variance
variance = pca.explained_variance_ratio_ * 100
cumulative = np.cumsum(variance)
plt.bar(range(1, len(variance)+1), variance, color="blue", alpha=0.7, label="Each component")
plt.plot(range(1, len(variance)+1), cumulative, color="red", marker="o", label="Cumulative")
plt.axhline(80, color="green", linestyle="--", label="80% threshold")
plt.xlabel("Component")
plt.ylabel("Variance Explained (%)")
plt.title("PCA: Variance Explained per Component")
plt.legend()
plt.savefig("Images/Variance.png")
plt.show()