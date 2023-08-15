#CLUSTERING NEW TRY
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math
from random import *
import sys
from numpy import arange

#GET DATA

path=open('D:\IITM MTech\FIrst Semester\Pattern Recognition and Machine Learning\Assignment\PRML Assignment 1\PRML Assignment 1/Dataset_test.csv')
data=np.loadtxt(path,delimiter=",",dtype='float')

k=5

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
        if(min_distance_cluster!=-1):
            assignment_list[i]=min_distance_cluster

#INITIALIZING THE OBJECTIVE VALUE
objective_f_val=find_objective_function_value()

print("THE DATA: ",data)
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
    if(find_objective_function_value()>=objective_f_val):
        print("ALGORITHM CONVERGED IN STEP NO: ",step)
        print("THE NEW VALUE OF ERROR IS: ",find_objective_function_value())
        print("FINAL MEANS: ",old_mean)
        print("FINAL ASSIGNMENT MATRIX: ",old_z)
        print("FINAL OBJECTIVE VALUE: ",objective_f_val)
        break
    else:
        objective_f_val=find_objective_function_value()
color=["green","red","blue","yellow","pink"]

x_comp=[data[i][0] for i in range(len(data))]
y_comp=[data[i][1] for i in range(len(data))]
colour_name=[color[assignment_list[i]-1] for i in range(len(data))]
plt.scatter(x_comp, y_comp, color="black")



#VORONOI REGION PLOTTING
#FIND COLOR
def find_color(i,j):
    dist=10000000
    ans=k
    for x in range(0,k):
        t=(i-means[x][0])**2 + (j-means[x][1])**2
        #print(t)
        if(dist>t):
            dist=t
            ans=x
    return(ans)

#CREATE THE POINTS TO BE PLOTTED
x_component_voronoi=[]
y_component_voronoi=[]
color_voronoi=[]

available_color=["green","red","blue","yellow","pink"]

for i in arange(-10,10.001,0.1):
    for j in arange(-10,10.001,0.1):
        x_component_voronoi.append(i)
        y_component_voronoi.append(j)
        color_voronoi.append(find_color(i,j))
color_voronoi_text=[available_color[color_voronoi[i]] for i in range(len(color_voronoi))]
plt.scatter(x_component_voronoi, y_component_voronoi, color=color_voronoi_text)

#PLOT THE MEANS
x_means=[means[i][0] for i in range(k)]
y_means=[means[i][1] for i in range(k)]
plt.scatter(x_means, y_means, color="black")

# x-axis label
plt.xlabel('X Coordinate')
# frequency label
plt.ylabel('Y Coordinate')
# plot title
plt.title('Voronoi regions with their respective Means')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()

#PRINT THE FINAL MEANS REPREENTING EACH CLUSTER
for i in range(len(means)):
    print("Mean of cluster number ",i+1," is (",means[i][0],",",means[i][1],") -",available_color[i]," colour")
    


        
