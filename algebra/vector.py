import numpy as np 

A = np.array([[1,2,3],[4,5,6],[7,8,9]]) #This is a matrix
B = np.array ([10,11,12]) #this is a vector
C = np.array ([13,14,15])
dot_product = np.dot(B,C)

print ("The dot product is: ", dot_product)
