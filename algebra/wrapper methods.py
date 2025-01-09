from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 


X = [[10,50,30,1],[20,50,40,1],[30,40,50,1],[40,50,60,1]]
y = [1,0,1,0] #target

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(X,y)
print("Selected Features", fit.support_)