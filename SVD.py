import numpy as np

A = np.asarray([[-5,  -5,  -1,  0,  0,  0,  500,  500,  100],
                [0,  0,  0,  -5,  -5,  -1,  500,  500,  100],
                [-150,  -5,  -1,  0,  0,  0,  30000,  1000,  200],
                [0,  0,  0,  -150,  -5,  -1,  12000,  400,  80],
                [-150,  -150,  -1,  0,  0,  0,  33000,  33000,  220],
                [0,  0,  0,  -150,  -150,  -1,  12000,  12000,  80],
                [-5,  -150,  -1,  0,  0,  0,  500,  15000,  100],
                [0,  0,  0,  -5,  -150,  -1,  1000,  30000,  200]])

# A = U*sig*Vt

U_A = A.dot(A.T)
V_A = A.T.dot(A)

# U is eigen vector of A.dot(A.T)
# V is eigen vector of A.T.dot(A)

### Calculating SVD ######

U = np.linalg.eig(U_A)[1]
print('U Matrix is: ')
print(U)

V = np.linalg.eig(V_A)[1]
print('V Matrix is: ')
print(V)

sigma = np.sqrt(np.absolute(np.linalg.eig(V_A)[0]))
S = np.diag(sigma)
S = S[0:8, :]
print('Sigma Matrix is:')
print(S)

###### Homography #######
H = V[:, 8]
H = np.reshape(H,(3,3))
print('H matrix is: ')
print(H)