"""
The dataset contains the following columns:

OrderID: Unique identifier for each transaction.
Product: The name of the product sold (e.g., Notebook, Pen, Marker).
Region: The region where the product was sold (e.g., North, South, East, West).
Month: The month of the transaction (numeric, 1–12).
Price: The price of a single unit of the product.
Quantity: The number of units sold in the transaction.
Discount: The discount applied to the transaction (in decimal, e.g., 0.05 for 5% discount).
Date: The date of the transaction.

STEPS:
Load the Dataset
Use pandas to read the dataset sales_data.csv into a DataFrame.

Data Inspection
Display the first 5 rows of the dataset to understand its structure.

Calculate Total Revenue
Add a new column named Revenue that calculates the total revenue for each transaction using the formula:

Revenue=Price×Quantity×(1−Discount)
Descriptive Statistics
Use Pandas methods to calculate the following statistics for the Revenue column:

Mean
Median
Standard deviation
Minimum and maximum values for the Quantity column.
Group Data by Region
Group the dataset by the Region column and calculate the average Revenue for each region.

Correlation Analysis
Compute the correlation matrix for all numerical columns in the dataset.

Pivot Table
Create a pivot table that shows the average Revenue for each combination of Region and Product.

Filtering Data
Filter the dataset to show only rows where:

The Product is "Pen".
The Discount is greater than 0.
Export the Enriched Dataset
Export the modified dataset (with the new Revenue column) to a CSV file named sales_data_enriched.csv.

Final Questions (Challenge)

Calculate the total Revenue for each Month.
Identify the most sold Product in terms of Quantity.
Determine the correlation between Price and Quantity.
Calculate the variance of the Revenue column.
Identify the Region with the highest total Revenue.
"""
import pandas as pd
import numpy as np

# Complete script for analyzing the dataset "sales_data.csv"
# Ensure the file "sales_data.csv" is in the same directory as this script.

# 1) Load the dataset
df = pd.read_csv('sales_data.csv')

# 2) Inspect the first 5 rows
print("First 5 records of the dataset:")
print(df.head(), "\n")

# 3) Calculate the 'Revenue' column
df['Revenue'] = df['Price'] * df['Quantity'] * (1 - df['Discount'])
print("Dataset with the 'Revenue' column:")
print(df.head(), "\n")

# 4) Descriptive statistics
mean_revenue = df['Revenue'].mean()
median_revenue = df['Revenue'].median()
std_revenue = df['Revenue'].std()
min_quantity = df['Quantity'].min()
max_quantity = df['Quantity'].max()

print("Descriptive Statistics for Revenue:")
print(f" - Mean  : {mean_revenue}")
print(f" - Median: {median_revenue}")
print(f" - Std Dev: {std_revenue}\n")
print("Minimum and Maximum values for Quantity:")
print(f" - Min: {min_quantity}")
print(f" - Max: {max_quantity}\n")

# 5) Group data by 'Region' and calculate average Revenue
region_revenue_mean = df.groupby('Region')['Revenue'].mean()
print("Average Revenue by 'Region':")
print(region_revenue_mean, "\n")

# 6) Correlation matrix
corr_matrix = df.select_dtypes(include=['number']).corr()
print("Correlation matrix:")
print(corr_matrix, "\n")

# 7) Pivot Table (Average Revenue by 'Region' and 'Product')
pivot_rev = pd.pivot_table(df, values='Revenue', index='Region', columns='Product', aggfunc='mean')
print("Pivot Table - Average Revenue by (Region, Product):")
print(pivot_rev, "\n")

# 8) Filter data: Product == "Pen" and Discount > 0
filtered_df = df[(df['Product'] == 'Pen') & (df['Discount'] > 0)]
print("Filtered rows (Product='Pen' and Discount>0):")
print(filtered_df, "\n")

# 9) Export the enriched DataFrame
df.to_csv('sales_data_enriched.csv', index=False)
print("File 'sales_data_enriched.csv' exported successfully.\n")

# 10) Final questions (Challenge)

# 10.a) Calculate total Revenue for each Month
monthly_revenue = df.groupby('Month')['Revenue'].sum()
print("Total Revenue for each Month:")
print(monthly_revenue, "\n")

# 10.b) Most sold product in terms of Quantity
product_quantity_sum = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
most_sold_product = product_quantity_sum.index[0]
print("Total Quantity sold for each Product:")
print(product_quantity_sum, "\n")
print(f"Most sold Product (Quantity): {most_sold_product}\n")

# 10.c) Correlation between Price and Quantity
price_quantity_corr = df[['Price', 'Quantity']].corr()
print("Correlation between 'Price' and 'Quantity':")
print(price_quantity_corr, "\n")

# 10.d) Variance of the Revenue column
revenue_variance = df['Revenue'].var()
print(f"Variance of 'Revenue': {revenue_variance}\n")

# 10.e) Region with the highest total Revenue
region_revenue_sum = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
top_region = region_revenue_sum.index[0]
print("Total Revenue for each Region:")
print(region_revenue_sum, "\n")
print(f"Region with the highest total Revenue: {top_region}")

