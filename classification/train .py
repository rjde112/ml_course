import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import joblib  # Import joblib to save the model
import pandas as pd  # Import pandas to manage CSV saving

# 1 Load the handwritten digits dataset
digits = datasets.load_digits()
X = digits.data  # dataset features (images)
y = digits.target  # labels (digits from 0 to 9)

# 2 Split into training and test sets (80% train and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Normalized training data:\n", X_train)
print("Normalized test data:\n", X_test)

# Visualize some images from the dataset
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.suptitle("Examples of handwritten digits in the dataset")
plt.show()

# Train the KNN model
# Use GridSearch to find the best hyperparameter
param_grid = {'n_neighbors': np.arange(1, 100)}
knn = KNeighborsClassifier()
knn_gs = GridSearchCV(knn, param_grid, cv=5)
knn_gs.fit(X_train, y_train)

# Best hyperparameter found
print(f"Best number of neighbors: {knn_gs.best_params_['n_neighbors']}")

# Train the model with the best hyperparameter set
knn_best = KNeighborsClassifier(n_neighbors=knn_gs.best_params_['n_neighbors'])
knn_best.fit(X_train, y_train)

# 5 Prediction on the test data
y_pred = knn_best.predict(X_test)

# 6 Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Model accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion matrix heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 7 Save the model in pkl format using joblib
model_filename = 'knn_digits_model.pkl'
joblib.dump(knn_best, model_filename)
print(f"Model saved as {model_filename}")

# 8 Create a DataFrame for training data and save to CSV
df_train = pd.DataFrame(X_train, columns=[f'Feature_{i}' for i in range(X_train.shape[1])])
df_train['Target'] = y_train  # Add the labels column

# Save the training dataset to a CSV file
csv_filename = 'training_dataset.csv'
df_train.to_csv(csv_filename, index=False)
print(f"Training dataset saved as {csv_filename}")
