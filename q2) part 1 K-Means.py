#CLUSTERING NEW TRY
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from random import *
import sys


#GET DATA
path=open('D:\IITM MTech\FIrst Semester\Pattern Recognition and Machine Learning\Assignment\PRML Assignment 1\PRML Assignment 1/Dataset_test.csv')
data=np.loadtxt(path,delimiter=",",dtype='float')

k=4

pick_mean_from=[i for i in range(len(data))]




#INITIALISE MEANS
means=[]

for i in range(k):
    index=randint(0, len(pick_mean_from)-1)
    means.append(data[pick_mean_from[index]])
    del pick_mean_from[index]

#INITIALIZE ASSIGNMENT LIST
assignment_list=[randint(1, k) for i in range(len(data))]


#The space in this matrix is used to calculate difference of each point wrt all available means, including its own mean
#This is will be declared inside
distance_to_means=[0 for i in range(k)]


#THIS FUNCTION WILL CALCULATE THE NEW MEANS GIVEN A CHANGE IN ASSIGNMENT LIST
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
    return
def find_objective_function_value():
    ans=0
    for i in range(len(data)):
        t=data[i]-means[assignment_list[i]-1]
        t=t@t.T
        ans+=t
    return(ans)

def k_means():
    for i in range(len(data)):
        difference=[0 for i in range(k)]
        for j in range(k):
            t=data[i]-means[j]
            #print("THE VALUE OF J IS: ",j,"THE VALUE OF DATA[i] is:",data[i]," The value of mean[j] ",means[j]," THE VALUE OF T IS :",t)
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

#print("THE DATA: ",data)
print("INITIALLY THE VALUE OF MEANS= ",means)
print("INITIALLY THE VALUE OF ASSIGNMENT LIST: ",assignment_list)
print("INITIALLY THE OBJECTIVE VALUE: ",objective_f_val)

#STORE ERROR FUNCTION VALUE
error_func=[]
error_func.append(objective_f_val)
Iteration=[]
step=0
Iteration.append(step)
while(1):
    old_mean=means[:]
    old_z=assignment_list[:]
    find_means()
    k_means()
    step+=1
    error_func.append(find_objective_function_value())
    Iteration.append(step)
    if(find_objective_function_value()>=objective_f_val):
        print("ALGORITHM CONVERGED IN STEP NO: ",step)
        print("THE NEW VALUE OF ERROR IS: ",find_objective_function_value())
        print("FINAL MEANS: ",old_mean)
        print("FINAL ASSIGNMENT MATRIX: ",old_z)
        print("FINAL OBJECTIVE VALUE: ",objective_f_val)
        break
    else:
        objective_f_val=find_objective_function_value()

# plotting the line 1 points 
plt.plot(Iteration, error_func, label = "Error Function")
  
  
# naming the x axis
plt.xlabel('Iteration Number')
# naming the y axis
plt.ylabel('Error Function Value')
# giving a title to my graph
plt.title('Error Function vs Iteration Number')
  
# show a legend on the plot
plt.legend()
  
# function to show the plot
plt.show()


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
plt.title('K-Means Clustering ')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()


        
