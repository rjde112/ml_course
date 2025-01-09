from sklearn.feature_selection import SelectKBest, f_classif

X = [[10,50,30],[20,50,40],[30,40,50],[40,50,60]]
y = [1,0,1,0] #target

#select top 2 features
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X,y)

print("matrix: \n", X)
print("\nSelected features: \n", X_new)