import pandas as pd



# Load the cleaned data
df = pd.read_csv("gameandgrade_cleaned.csv")

# Discretize Grade into letter grades
df["Grade Category"] = pd.cut(df["Grade"],
                               bins=[0, 50, 60, 70, 80, 90, 101],
                               labels=["F", "D", "C", "B", "A", "A+"])

# Show results
print("Grade Discretization:")
print(df["Grade Category"].value_counts().sort_index())