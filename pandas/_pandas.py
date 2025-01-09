import pandas as pd


df = pd.read_csv("/home/trace/training/ml_course/pandas/dataset.csv")

#2 print the intial dataframe
print("initial dataframe:")
print(df.head())

#3 Data Cleaning: Handling missing values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Performance Score"] = df["Performance Score"].fillna(0)
df["Age"] = df["Age"].fillna(df["Age"].median())

print("\nDataframe after handling missing values:")
print(df.head())


