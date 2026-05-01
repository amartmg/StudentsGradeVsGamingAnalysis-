import pandas as pd

# Load the CSV file into a dataframe
df = pd.read_csv("gameandgrade.csv")

# Show the first 5 rows
print(df.head())

# Show how many rows and columns we have
print(df.shape)

# Check for missing values in each column
print(df.isnull().sum())

# Remove extra hidden spaces from column names
df.columns = df.columns.str.strip()

# Rename NSex to Sex
df = df.rename(columns={"NSex": "Sex"})

# Print the column names to verify
print(df.columns)

# Check the data type of each column
print(df.dtypes)

# Fix typo in Grade column, for example 92..00 -> 92.00
df["Grade"] = df["Grade"].str.replace("..", ".", regex=False)

# Remove any letters from Grade column, for example 88.00N -> 88.00
df["Grade"] = df["Grade"].str.replace(r"[a-zA-Z]", "", regex=True)

# Convert Grade from text to number
df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")

# Remove rows where Grade could not be converted
df = df.dropna(subset=["Grade"])

# Confirm Grade is now a number
print(df["Grade"].dtype)

# Save the cleaned data to a new CSV file
df.to_csv("gameandgrade_cleaned.csv", index=False)

print("Cleaned file saved!")