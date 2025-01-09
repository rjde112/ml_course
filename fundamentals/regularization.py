from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

#L1 Regularization (Lasso)
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

model_l1.fit(X,y)
print("L1 Coefficients: ", model_l1.coef_)

#L2 Regularization (Ridge)
model_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.1)

model_l2.fit(X,y)
print("L2 Coefficients: ", model_l2.coef_)

