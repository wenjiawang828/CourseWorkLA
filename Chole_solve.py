# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

path = 'q3b.txt'
with open(path) as f:
    data = np.loadtxt(path,dtype='float32')
# D is a diagonal positive definite matrix
# The first column of the file lists the diagonal entries of D;
# Columns 2 and 3 give the two columns of A;    A's dimension = n*2
# Column 4 gives the vector b
D = np.diag(data[:,0])
A = data[:,1:3]
b = data[:,3]
# M is a symmetric positive definite matrix M=D+AA^T
M = D+np.dot(A,A.T)
# M = R^T*R
R = np.triu(M)
# #---------------------------------#
# # Cholesky decomposition (writen by hand)
# #---------------------------------#
n = np.size(A,0)
# for i in range(n):
#     for j in range(i+1,n):
#         for k in range(j,n):
#             R[j,k]=R[j,k]-R[i,k]*R[i,j]/R[i,i]

#         R[i,j]=R[i,j]/R[i,i]**0.5;

#     R[i,i]=R[i,i]**0.5
# #---------------------------------#
# # Cholesky decomposition (from np)
L = np.linalg.cholesky(M)
R = L.T
# #---------------------------------#
# Mx=b, R^T*Rx=b
# forward Ly=b, L=R^T, y=Rx
b[0]=b[0]/R[0,0]
for i in range(1,n):
    for j in range(0,i):
        b[i]=b[i]-R[j,i]*b[j]
    b[i]=b[i]/R[i,i]
# backward Ux=y, R=U
b[n-1]=b[n-1]/R[n-1,n-1]
for i in range(n-2,-1):
    for j in range(i+1,n+1):
        b[i]=b[i]-R[i,j]*b[j]
    b[i]=b[i]/R[i,i]

print(b[-1,-11])