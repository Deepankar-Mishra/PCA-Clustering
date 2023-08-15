#Q1-Part i)
import numpy as np
from numpy.linalg import eig

path=open('D:\IITM MTech\FIrst Semester\Pattern Recognition and Machine Learning\Assignment\PRML Assignment 1\PRML Assignment 1/Dataset_test.csv')
data=np.loadtxt(path,delimiter=",",dtype='float')

#CENTERING
m=len(data)

mean=np.array([0,0])

for i in range(m):
    mean=np.add(mean,data[i])

mean=(1/m)*mean

for i in range(m):
    data[i]=np.subtract(data[i],mean)

#Construct Covariance Matrix

x=np.array([[0,0],[0,0]])

for i in range(m):
    x=x+np.array([data[i]]).T@np.array([data[i]])

x=x/m

#Eigendecomposition

e,w=eig(x)
w=w.T

#Sorting the eigenvector based on their corresponding eigenvalues
e_w=[[e[i],w[i]] for i in range(len(e))]
e_w.sort(key=lambda h:h[0],reverse=1)

#Finding the variance represented by each eigenvector, by dividing its correponding eigenvalue by sum of all eigenvalues.
sum_of_eigenvalues=0
for i in range(len(e)):
    sum_of_eigenvalues+=e_w[i][0]
for i in range(len(e)):
    print('The principal component number ',i+1,'is: ',e_w[i][1]," The variance represented by this principal component is: ",(e_w[i][0]*100)/sum_of_eigenvalues)
