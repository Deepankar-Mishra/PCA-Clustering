#SPECTRAL CLUSTERING

#EXPONENTIAL KERNEL PCA

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from random import randint

path=open('D:\IITM MTech\FIrst Semester\Pattern Recognition and Machine Learning\Assignment\PRML Assignment 1\PRML Assignment 1/Dataset_test.csv')
data=np.loadtxt(path,delimiter=",",dtype='float')

#Define KERNEL
def kernel_polynomial(x,y):
    ans=1+x@y.T
    factor=1+x@y.T
    for i in range(2,d+1):
        ans=ans*factor
    return(ans)
def kernel_exponential(x,y,sigma):
    p=(y-x)@(x-y).T
    p=p/sigma
    p=p/sigma
    p=p/2
    ans=float((math.e)**p)
    return(ans)
#CONSTRUCT HIGH DIMENSIONAL COVARIANCE MATRIX

x=[[0 for i in range(1000)] for j in range(1000)]

d=3
cnt=0
for i in range(1000):
    for j in range(1000):
        cnt+=1
        x[i][j]=kernel_exponential(data[i],data[j],0.1)
x=np.array(x)

e,w=eig(x)

for i in range(1000):
    e[i]=e[i].real
    
w=w.T

e_w=[[e[i],i] for i in range(len(e))]
e_w.sort(key=lambda h:h[0],reverse=1)
#DEFINE NUMBER OF CLUSTERS
k=4

h=np.array([w[e_w[i][1]] for i in range(k)])
for i in range(len(h)):
    for j in range(len(h[0])):
        h[i][j]=h[i][j].real
h=h.T

h_before_normalisation=h[:]


assignment_list=[0 for i in range(len(data))]
#print(type(h[0][2]))
for i in range(len(h)):
    temp=[[h[i][j].real,j] for j in range(len(h[i]))]
    temp.sort()
    assignment_list[i]=temp[-1][1]


color=["green","red","blue","yellow"]

x_comp=[data[i][0] for i in range(len(data))]
y_comp=[data[i][1] for i in range(len(data))]
colour_name=[color[assignment_list[i]-1] for i in range(len(data))]
plt.scatter(x_comp, y_comp, color=colour_name)


# x-axis label
plt.xlabel('X Coordinate')
# frequency label
plt.ylabel('Y Coordinate')
# plot title
plt.title('Clustering (Assign Cluster, seeing max in row)')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()

means=[0 for i in range(k)]

def find_means():
    cnt=[0 for i in range(k)]
    new_means=[0 for i in range(k)]
    for i in range(len(data)):
        cluster_no=assignment_list[i]-1
        new_means[cluster_no]+=data[i]
        cnt[cluster_no]+=1
    for i in range(len(means)):
        if(cnt[i]>0):
            new_means[i]=new_means[i]/cnt[i]
            means[i]=new_means[i]
find_means()


