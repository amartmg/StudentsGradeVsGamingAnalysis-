# Games and Success in Students : A Data Mining Project

A data mining project analyzing the relationship between students' gaming habits and academic performance using the **GameAndGrade dataset**.

## Dataset
- **Records:** 770 students
- **Features:** 10 attributes (gaming habits, parental education, grade)
- **Target:** Grade (33–100)

## Project Structure

| File | Description |
|------|-------------|
| `gameandgrade.csv` | Original dataset |
| `gameandgrade_cleaned.csv` | Cleaned dataset |
| `cleaning.py` | Data cleaning |
| `reduction.py` | Data reduction (PCA, Regression, Clustering, Sampling) |
| `normalization.py` | Data normalization (Min-Max, Z-Score) |
| `discrettization.py` | Data discretization (Grade categories) |
| `Classification.py` | KNN Classification model |
| `cluster_analysis.py` | K-Means Cluster Analysis model |

## Data Preprocessing Steps

### 1. Data Cleaning
- Fixed typos in Grade column (`92..00` → `92.00`, `88.00N` → `88.00`)
- Renamed `NSex` column to `Sex`
- No missing values found

### 2. Data Reduction
- **Attribute Subset Selection:** Removed `Playing Often` (correlation = 0.009)
- **Linear Regression:** Mother Education is the best single predictor (R²=0.1245)
- **Multiple Regression:** All features combined R²=0.198
- **Histogram Analysis:** Grade distribution is left-skewed
- **Clustering:** K-Means grouped students into 3 profiles
- **Sampling:** 100-student sample matches full dataset average
- **PCA:** 5 components explain 84.66% of variance

### 3. Normalization
- Min-Max Normalization: scales values to [0, 1]
- Z-Score Standardization: scales to mean=0, std=1

### 4. Discretization
- Grade → Letter categories (F / D / C / B / A / A+)
- Playing Hours → None / Low / Medium / High

## Models Used

### KNN Classification
- Predicts student grade category using K-Nearest Neighbors
- Best K = 7 with accuracy ~37%
- 6 classes: F, D, C, B, A, A+

### K-Means Cluster Analysis
- Groups students into 3 natural clusters
- **Cluster 0:** Moderate Gamers (avg Grade 75.70)
- **Cluster 1:** Heavy Gamers (avg Grade 76.07)
- **Cluster 2:** Non-Gamers (avg Grade 81.51)

## Key Findings
- Mother Education (r=0.35) and Father Education (r=0.32) are the strongest predictors of Grade
- More gaming hours = lower grades confirmed by both models
- Non-gaming students achieve the highest average grade (81.51)

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## How to Run
1. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Place `gameandgrade.csv` in the same folder as the scripts
3. Run scripts in this order:
   - `cleaning.py` first (generates `gameandgrade_cleaned.csv`)
   - Then any other script in any order 
