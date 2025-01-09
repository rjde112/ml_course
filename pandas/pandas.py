import pandas as pd 
import numpy as np 

#1 load the CSV
df = pd.read_csv("dataset.csv")

#2 print the intial dataframe
print("initial dataframe:")
print(df.head())

#3 Data Cleaning: Handling missing values
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Performance Score"] = df["Performance Score"].fillna(0)
df["Age"] = df["Age"].fillna(df["Age"].median())

print("\nDataframe after handling missing values:")
print(df.head())


