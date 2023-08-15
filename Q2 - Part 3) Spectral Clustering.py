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
        x[i][j]=kernel_exponential(data[i],data[j],2)
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

for i in range(len(h)):
    #print("For i= ",i)
    norm_val=np.linalg.norm(h[i])
    #print("THE VALUE OF H[i] before norm: ",h[i])
    if(norm_val!=0):
        h[i]=(1/norm_val)*h[i]
    #print("THE VALUE OF H[i] after norm: ",h[i])
h_after_normalisation=h[:]
#h=h.T
new_data=np.array([h[i] for i in range(len(h))])
for i in range(len(new_data)):
    for j in range(len(new_data[0])):
        new_data[i][j]=new_data[i][j].real

print("new data= ",new_data)

#CLUSTERING ALGO

pick_mean_from=[i for i in range(len(new_data))]




#INITIALISE MEANS
means=[]

for i in range(k):
    index=randint(0, len(pick_mean_from)-1)
    means.append(new_data[pick_mean_from[index]])
    del pick_mean_from[index]


#INITIALIZE ASSIGNMENT LIST
assignment_list=[randint(1, k) for i in range(len(new_data))]


#The space in this matrix is used to calculate difference of each point wrt all available means, including its own mean
#This is will be declared inside
distance_to_means=[0 for i in range(k)]


#THIS FUNCTION WILL CALCULATE THE NEW MEANS GIVEN A CHANGE IN ASSIGNMENT LIST
def find_means():
    cnt=[0 for i in range(k)]
    new_means=[0 for i in range(k)]
    for i in range(len(new_data)):
        cluster_no=assignment_list[i]-1
        new_means[cluster_no]+=new_data[i]
        cnt[cluster_no]+=1
    for i in range(len(means)):
        if(cnt[i]>0):
            new_means[i]=new_means[i]/cnt[i]
            means[i]=new_means[i]
    return
def find_objective_function_value():
    ans=0
    for i in range(len(new_data)):
        try:
            t=new_data[i]-means[assignment_list[i]-1]
            t=t@t.T
            ans+=t
        except:
            pass
    return(ans)

def k_means():
    for i in range(len(new_data)):
        difference=[0 for i in range(k)]
        for j in range(k):
            t=new_data[i]-means[j]
            difference[j]=t@t.T
        #IF DISTANCE OF POINT TO ITS OWN MEAN> DISTANCE TO ANOTHER MEAN . REASSIGNMENT
        dist_to_own_mean=difference[assignment_list[i]-1]
        #FIND THE MIN(EXCEPT DISTANCE TO ITS OWN)
        min_distance_cluster=-1
        min_distance=dist_to_own_mean
        for j in range(k):
            if(j!=assignment_list[i]-1):
                if(min_distance>difference[j]):
                    min_distance_cluster=j+1
                    min_distance=difference[j]
        #REASSIGN IF min_distance_cluster!=-1
        #print("For point: ",data[i])
        #print(difference)
        if(min_distance_cluster!=-1):
            assignment_list[i]=min_distance_cluster


#INITIALIZING THE OBJECTIVE VALUE
objective_f_val=find_objective_function_value()

print("THE DATA: ",new_data)
print("INITIALLY THE VALUE OF MEANS= ",means)
print("INITIALLY THE VALUE OF ASSIGNMENT LIST: ",assignment_list)
print("INITIALLY THE OBJECTIVE VALUE: ",objective_f_val)


step=0
while(1):
    old_mean=means[:]
    old_z=assignment_list[:]
    find_means()
    k_means()
    step+=1
    #print("THE DATA: ",data)
    #print("\nAFTER STEP NUMBER: ",step)
    #print("THE VALUE OF MEANS= ",means)
    #print("THE VALUE OF ASSIGNMENT LIST: ",assignment_list)
    #print("THE OBJECTIVE VALUE: ",objective_f_val)
    if(find_objective_function_value()>=objective_f_val):
        print("ALGORITHM CONVERGED IN STEP NO: ",step)
        print("THE NEW VALUE OF ERROR IS: ",find_objective_function_value())
        print("FINAL MEANS: ",old_mean)
        print("FINAL ASSIGNMENT MATRIX: ",old_z)
        print("FINAL OBJECTIVE VALUE: ",objective_f_val)
        break
    else:
        objective_f_val=find_objective_function_value()
color=["green","red","blue","yellow"]

x_comp=[data[i][0] for i in range(len(new_data))]
y_comp=[data[i][1] for i in range(len(new_data))]
colour_name=[color[assignment_list[i]-1] for i in range(len(new_data))]
plt.scatter(x_comp, y_comp, color=colour_name)


# x-axis label
plt.xlabel('X Coordinate')
# frequency label
plt.ylabel('Y Coordinate')
# plot title
plt.title('Spectral Clustering')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()

