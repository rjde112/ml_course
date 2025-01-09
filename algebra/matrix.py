import numpy as np 

#Create a 3x3 matrix
matrix = np.zeros((3,3))
print("Initial Matrix:")
print (matrix)

matrix[1,1] = 5
print("\nModified Matrix")
print (matrix)

matrix.fill(0)
print("\nReset Matrix")
print (matrix)
