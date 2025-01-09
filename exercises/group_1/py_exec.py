import pandas as pd

# 1. Load training data
print("\nOpen dataset:")
with open("train_dataset.csv", "r") as file:
    df = pd.read_csv(file)

#2. Display top 5 rows
print("\nDiscplay top 5 rows:")
df_new = df.head(5)
print(df_new)

#3. Calculate Revenue
df_revenue = df
df_revenue["Revenue"] = (df["Price"].astype(float) * df["Quantity"].astype(float))*(1 - df["Discount"].astype(float))
print(df_revenue)

#4. Statistical Information 
print(f"Mean Salary: {df_revenue['Revenue'].mean()}")
print(f"Median Salary: {df_revenue['Revenue'].median()}")
print(f"Salary Standard Deviation: {df_revenue['Revenue'].std()}")
print(f"Min: {df_revenue['Quantity'].min()}")
print(f"Max: {df_revenue['Quantity'].max()}")

#5. Grouping by Region
grouped_region = df_revenue.groupby("Region")["Revenue"].mean()
print("\nAverage Revenue by Region:")
print(grouped_region)

#6. Correlation of numeric values
df_numeric = df_revenue.drop(["OrderID", "Month"], axis=1)
df_numeric = df_numeric.select_dtypes(include='number')  
print("\nCorrelation Matrix (only numeric columns):")
print(df_numeric.corr())

#7. Pivot Tables by Region and Product
pivot_table = df_revenue.pivot_table(values="Revenue", index="Region", columns="Product", aggfunc="mean")
print("\nPivot Table (Average Revenue by Region and Product):")
print(pivot_table)

#8. Data Filter
print("\nData Filtered on Pen")
df_data_filter = df_revenue[(df_revenue["Product"] == "Pen") & (df_revenue["Quantity"] >= 0)]
print(df_data_filter)

#9. Data Filter
print("\nNew dataset export")
df_revenue.to_csv("sales_data_enriched.csv", index=False)

#10. Challenge questions
##Average Revenue
grouped_month = df_revenue.groupby("Month")["Revenue"].mean()
print("\nAverage Revenue by Month:")
print(grouped_month)

##Max Product
grouped_product = df_revenue.groupby("Product")["Quantity"].sum()
max_product = grouped_product.sort_values(ascending=False).head(1)
print(f"\nMax product: \n{max_product}")

##Correlation Price & Quantity
df_price_quantity = df_revenue['Price'].corr(df_revenue['Quantity'])
print("\nCorrelation Matrix (Price & Quanity):")
print(df_price_quantity)

##Correlation Price & Quantity
df_price_quantity = df_revenue['Price'].corr(df_revenue['Quantity'])
print("\nCorrelation Matrix (Price & Quanity):")
print(df_price_quantity)

##Variance of Revenue
df_rev_variance = df_revenue['Revenue'].var()
print("\nVariation (Revenue):")
print(df_rev_variance)

##Max Revenue
grouped_rr = df_revenue.groupby("Region")["Revenue"].sum()
max_grouped_rr = grouped_rr.sort_values(ascending=False).head(1)
print(f"\nMax Revenue: \n{max_grouped_rr}")