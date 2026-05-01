import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the cleaned data
df = pd.read_csv("gameandgrade_cleaned.csv")

# ── 1. SELECT COLUMNS TO NORMALIZE ────────────────────────────────────────
# We normalize all feature columns (not Grade — that's our target)
feature_cols = ["Playing Years", "Playing Hours", "Playing Games",
                "Parent Revenue", "Father Education", "Mother Education"]

# ── 2. SHOW ORIGINAL VALUES ────────────────────────────────────────────────
print("BEFORE Normalization:")
print(df[feature_cols].head())
print("\nOriginal ranges:")
for col in feature_cols:
    print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

# ── 3. MIN-MAX NORMALIZATION ───────────────────────────────────────────────
# Scales all values to between 0 and 1
# Formula: (value - min) / (max - min)
# Used when we don't want any column to dominate due to larger numbers
mm_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[feature_cols] = mm_scaler.fit_transform(df[feature_cols])

print("\nAFTER Min-Max Normalization (all values between 0 and 1):")
print(df_minmax[feature_cols].head())

# ── 4. Z-SCORE STANDARDIZATION ────────────────────────────────────────────
# Scales all values to have mean=0 and standard deviation=1
# Formula: (value - mean) / std
# Used when data needs to follow a normal distribution (e.g. for PCA)
z_scaler = StandardScaler()
df_zscore = df.copy()
df_zscore[feature_cols] = z_scaler.fit_transform(df[feature_cols])

print("\nAFTER Z-Score Standardization (mean=0, std=1):")
print(df_zscore[feature_cols].head())

# ── 5. COMPARE RESULTS ─────────────────────────────────────────────────────
print("\nNormalization Comparison (Playing Hours):")
print(f"  Original:  min={df['Playing Hours'].min()}, max={df['Playing Hours'].max()}")
print(f"  Min-Max:   min={df_minmax['Playing Hours'].min():.2f}, max={df_minmax['Playing Hours'].max():.2f}")
print(f"  Z-Score:   min={df_zscore['Playing Hours'].min():.2f}, max={df_zscore['Playing Hours'].max():.2f}")

print("\nNormalization complete!")