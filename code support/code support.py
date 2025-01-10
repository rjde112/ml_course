import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load the Iris dataset as an example
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Print original dataset shape
print("Original dataset shape:", X.shape)

# 2. Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

print("\nStandardized Data (first 5 samples):")
print(X_standardized[:5])

# 3. Normalization
normalizer = MinMaxScaler(feature_range=(0, 1))
X_normalized = normalizer.fit_transform(X)

print("\nNormalized Data (first 5 samples):")
print(X_normalized[:5])

# 4. Principal Component Analysis (PCA)
# Apply PCA to reduce dimensions to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

print("\nPCA reduced data (first 5 samples):")
print(X_pca[:5])

# Variance explained by each component
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio by PCA components:", explained_variance)

# 5. Visualization of PCA results
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=iris.target_names[label])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid()
plt.show()
