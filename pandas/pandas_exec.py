import pandas as pd
import numpy as np

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)

# 1. Load the CSV using open() as an alternative
with open("dataset.csv", "r") as file:
    df = pd.read_csv(file)

# 2. Display the initial DataFrame
print("Initial DataFrame:")
print(df)

# 3. Data Cleaning: Handling Missing Values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())  # Fill NaN in Salary with mean
df["Performance Score"] = df["Performance Score"].fillna(0)  # Fill NaN in Performance Score with 0
df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill NaN in Age with median

print("\nDataFrame after handling missing values:")
print(df)

# 4. Basic Information
print("\nBasic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 5. Advanced Filtering: iloc for Index-Based Selection
print("\nUsing iloc for selecting rows 0 to 4 and specific columns:")
print(df.iloc[0:5, [0, 2, 4]])  # Rows 0-4, columns ID, Age, Salary

# 6. Adding New Columns
df["Experience (Years)"] = 2025 - pd.to_datetime(df["Joining Date"]).dt.year
print("\nDataFrame with new column (Experience):")
print(df.head())

# 7. Grouping and Aggregation
grouped = df.groupby("Department")["Salary"].mean()
print("\nAverage Salary by Department:")
print(grouped)

# 8. Concatenation: Combining DataFrames
df_new = df.head(3)  # Create a subset
df_combined = pd.concat([df_new, df_new], axis=0).reset_index(drop=True)
print("\nConcatenated DataFrame:")
print(df_combined)

# 9. DataFrame Creation
custom_data = {
    "ID": [101, 102],
    "Name": ["Zoe", "Yasmine"],
    "Salary": [70000, 80000],
    "Department": ["Finance", "HR"]
}
df_custom = pd.DataFrame(custom_data)
print("\nCustom DataFrame:")
print(df_custom)

# 10. Merge: Joining Two DataFrames
merged_df = pd.merge(df, df_custom, on="Department", how="inner")
print("\nMerged DataFrame (on Department):")
print(merged_df)

# 11. Data Casting
df["Performance Score"] = df["Performance Score"].astype(int)
print("\nDataFrame after casting Performance Score to int:")
print(df.dtypes)

# 12. Normalizing Columns
df["Normalized Salary"] = (df["Salary"] - df["Salary"].mean()) / df["Salary"].std()
print("\nDataFrame with Normalized Salary:")
print(df.head())

# 13. Statistical Operations
print("\nStatistical Summary:")
print(f"Mean Salary: {df['Salary'].mean()}")
print(f"Median Salary: {df['Salary'].median()}")
print(f"Salary Standard Deviation: {df['Salary'].std()}")

# 14. Correlation Analysis (seleziona solo le colonne numeriche)
df_numeric = df.select_dtypes(include='number')  # Filtra le colonne numeriche
print("\nCorrelation Matrix (only numeric columns):")
print(df_numeric.corr())

# 15. Pivot Tables
pivot_table = df.pivot_table(values="Salary", index="Department", columns="Gender", aggfunc="mean")
print("\nPivot Table (Average Salary by Department and Gender):")
print(pivot_table)

# 16. Handling Duplicates
df.loc[len(df)] = df.iloc[0]  # Add a duplicate row for demonstration
print("\nDataFrame with a duplicate row added:")
print(df.tail())
df = df.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df.tail())

# 17. Feature Selection: Dropping Irrelevant Columns
df_reduced = df.drop(["ID", "Name"], axis=1)
print("\nDataFrame after Dropping Irrelevant Columns:")
print(df_reduced.head())

# 18. Export Cleaned Data
df.to_csv("cleaned_large_dataset.csv", index=False)
print("\nCleaned DataFrame exported to 'cleaned_large_dataset.csv'")
