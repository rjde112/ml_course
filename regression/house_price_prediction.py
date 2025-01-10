import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import XGBRegressor from XGBoost
from xgboost import XGBRegressor

# 1. Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# 3. Create a Pipeline:
#    - StandardScaler: normalizes the data
#    - PCA: dimensionality reduction
#    - XGBRegressor: final model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('xgb', XGBRegressor(random_state=42))
])

# 4. Define the parameter grid for GridSearchCV
#    - Limit n_components to [1..8] (California Housing has 8 features).
#    - Include several XGBoost hyperparameters for tuning.
param_grid = {
    'pca__n_components': [8],
    'xgb__n_estimators': [100, 300, 500],
    'xgb__max_depth': [3, 6, 10],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

# 5. Set up GridSearchCV
#    - scoring='neg_mean_squared_error' to minimize MSE
#    - cv=2 for demonstration (you can increase this for more robust estimates)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=2,  # or more, e.g., 5
    n_jobs=-1,
    verbose=1
)

# 6. Train the GridSearch on the training set
grid_search.fit(X_train, y_train)

# 7. Retrieve the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Best parameters found: {best_params}")

# 8. Evaluate on the test set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation on Test Set ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 9. Plot Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--',
         label='Ideal Fit')
plt.title('XGBoost Predictions vs. Actual House Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# 10. Save the test dataset and predictions to CSV
df_test_results = X_test.copy()
df_test_results['Actual'] = y_test
df_test_results['Predicted'] = y_pred

csv_filename = "california_housing_results_xgb.csv"
df_test_results.to_csv(csv_filename, index=False)
print(f"\nDataset with results saved to: {csv_filename}")
