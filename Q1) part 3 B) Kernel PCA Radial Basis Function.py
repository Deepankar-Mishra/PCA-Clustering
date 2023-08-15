#EXPONENTIAL KERNEL PCA Q1) part 2, B)

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math

path=open('D:\IITM MTech\FIrst Semester\Pattern Recognition and Machine Learning\Assignment\PRML Assignment 1\PRML Assignment 1/Dataset_test.csv')
data=np.loadtxt(path,delimiter=",",dtype='float')

#print(data)



m=len(data)

mean=np.array([0,0])

for i in range(m):
    mean=np.add(mean,data[i])

mean=(1/m)*mean

def kernel_exponential(x,y,sigma):
    p=(y-x)@(x-y).T
    p=p/sigma
    p=p/sigma
    p=p/2
    #print("p is: ",p)
    ans=float((math.e)**p)
    #print("ANS: ",ans)
    return(ans)

#CONSTRUCT HIGH DIMENSIONAL COVARIANCE MATRIX

x=[[0 for i in range(1000)] for j in range(1000)]

sigma=1
cnt=0
for i in range(1000):
    for j in range(1000):
        cnt+=1
        #print("DATA[i]: ",data[i])
        #print("DATA[j]: ",data[j])
        x[i][j]=kernel_exponential(data[i],data[j],sigma)
        #print(x[i][j])
x=np.array(x)
ident=np.identity(1000)
i_n_matrix=np.array([[0.001 for i in range(1000)] for j in range(1000)])
c_matrix=ident-i_n_matrix

x=(c_matrix@x)@c_matrix



e,w=eig(x)

for i in range(1000):
    e[i]=e[i].real
w=w.T

e_w=[[e[i],i] for i in range(len(e))]
e_w.sort(key=lambda h:h[0],reverse=1)



sum_of_eigenvalues=0
for i in range(len(e)):
    sum_of_eigenvalues+=e_w[i][0]
for i in range(3):
    print('The principal component number ',i+1,'is: ',w[e_w[i][1]]," The variance represented by this principal component is: ",e_w[i][0]/sum_of_eigenvalues)

alpha=[]
for i in range(2):
    #alpha.append([w[e_w[i][1]][j]/(e_w[i][0]**0.5) for j in range(1000)])
    alpha.append(w[e_w[i][1]]/(e_w[i][0]**0.5))
        



component=[]
x_comp=[]
y_comp=[]

alpha1=w[e_w[0][1]]
alpha2=w[e_w[1][1]]
for i in range(1000):
    k=np.array([x[i]])
    x_val=alpha1@k.T
    y_val=alpha2@k.T
    x_comp.append(x_val[0].real)
    y_comp.append(y_val[0].real)
    #component.append([alpha1@k.T[0][0],alpha2@k.T[0][0]])

# plotting points as a scatter plot
plt.scatter(x_comp, y_comp, label= "stars", color= "green", 
            marker= "*", s=30)

  
# x-axis label
plt.xlabel('Component along Principal Component 1 (Corresponding to largest eigenvalue)')
# frequency label
plt.ylabel('Component along Principal Component 2 (Corresponding to 2nd Highest eigenvalue)')
# plot title
plt.title('Exponential kernel used with sigma='+str(sigma))
# showing legend
plt.legend()
  
# function to show the plot
plt.show()

