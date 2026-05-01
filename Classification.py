import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

os.makedirs("Images", exist_ok=True)

# Load the cleaned data
df = pd.read_csv("gameandgrade_cleaned.csv")



# ── 1. CREATE TARGET VARIABLE ──────────────────────────────────────────────
# Convert continuous Grade into letter grade categories
# This is required for classification (needs categories, not numbers)
df["Grade Category"] = pd.cut(df["Grade"],
                               bins=[0, 50, 60, 70, 80, 90, 101],
                               labels=["F", "D", "C", "B", "A", "A+"])

# Remove any rows where Grade Category could not be assigned
df = df.dropna(subset=["Grade Category"])

print("Class distribution:")
print(df["Grade Category"].value_counts().sort_index())

# ── 2. PREPARE FEATURES AND TARGET ────────────────────────────────────────
# Remove Grade (we used it to make categories)
# Remove Playing Often (correlation = 0.009, not useful)
# Remove Grade Category (this is what we're predicting)
X = df.drop(columns=["Grade", "Grade Category", "Playing Often"])
y = df["Grade Category"]

# ── 3. NORMALIZE FEATURES ─────────────────────────────────────────────────
# KMeans uses distance — normalization ensures all features are equally weighted
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. SPLIT DATA — 80% TRAINING, 20% TESTING ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 5. TRAIN KNN MODEL (K=5) ───────────────────────────────────────────────
# K=5 is the standard starting point
# We test other values in Step 7 to find the best K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ── 6. EVALUATE THE MODEL ─────────────────────────────────────────────────
y_pred = knn.predict(X_test)

print("\nKNN Classification Results (k=5):")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 7. FIND BEST K VALUE ───────────────────────────────────────────────────
# Test K from 1 to 20 and pick the one with highest accuracy
accuracies = []
k_range = range(1, 21)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

best_k = k_range[accuracies.index(max(accuracies))]
print(f"\nBest K: {best_k} with accuracy {max(accuracies) * 100:.2f}%")

# Plot accuracy vs K
plt.figure(figsize=(8, 4))
plt.plot(k_range, [a * 100 for a in accuracies], marker="o", color="blue")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy (%)")
plt.title("KNN: Accuracy vs K Value")
plt.legend()
plt.tight_layout()
plt.savefig("Images/KNN_Accuracy.png")
plt.show()

# ── 8. CONFUSION MATRIX ───────────────────────────────────────────────────
# Shows how many students were correctly/incorrectly classified
cm = confusion_matrix(y_test, y_pred, labels=["F", "D", "C", "B", "A", "A+"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["F", "D", "C", "B", "A", "A+"],
            yticklabels=["F", "D", "C", "B", "A", "A+"])
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("Images/KNN_Confusion_Matrix.png")
plt.show()

print("\nClassification complete!")
print("Charts saved: KNN_Accuracy.png, KNN_Confusion_Matrix.png")