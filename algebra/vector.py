import numpy as np 

#Array definition
u = np.array([3,4,5])
v = np.array([1,2,3])

#Vector sum
sum_vector = u + v
print("Sum (u - v):", sum_vector )

#Difference
diff_vectors = u - v
print("Difference (u - v):", diff_vectors)

#dot product
dot_product = np.dot(u,v)
print("Dot product:" , dot_product)

#Vector Product (cross product)
cross_product = np.cross(u,v)
print ("Cross product (u x v)", cross_product)

#vector norm
norm_u = np.linalg.norm(u)
print("norm of (|u|):", norm_u)

#vector normalization
normalized_u = u / norm_u
print("Normalized vector u:", normalized_u)


