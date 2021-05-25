#!/usr/bin/env python
# coding: utf-8

# In[22]:


# City names alphabetically
cityNames = ['Винница','Днепр','Житомир','Запорожье','Ивано-Франковск','Киев','Кропивницкий','Луцк','Львов']
#,'Николаев','Одесса','Полтава','Ровно','Севастополь','Симферополь','Сумы','Тернополь','Ужгород','Харьков','Херсон','Хмельницкий','Черкассы','Чернигов','Черновцы'
# 'Донецк','Луганск',
print(cityNames)
N = len(cityNames)
print("Number of cities: ",N)


# In[23]:


# Distance matrix using maps.google.com. Last included city is 'Луцк'
d = [
    [  0, 588, 128, 651, 368, 268, 322, 389, 364],
    [588,   0, 596,85.4, 962, 480, 247, 875, 957],
    [128, 596,   0, 690, 413, 140, 392, 261, 402],
    [651,85.4, 690,   0,1026, 538, 310, 952,1021],
    [368, 962, 413,1026,   0, 560, 694, 263, 132],
    [268, 480, 140, 538, 560,   0, 303, 400, 541],
    [322, 247, 392, 310, 694, 303,   0, 670, 693],
    [389, 875, 261, 952, 263, 400, 670,   0, 180],
    [364, 957, 402,1021, 132, 541, 693, 180,   0]
    ]
#print(d)

# Time by car
t = [
    [  0, 484,  99, 521, 336, 236, 266, 311, 323],
    [484,   0, 474,  82, 866, 407, 234, 706, 765],
    [ 99, 474,   0, 582, 396, 101, 352, 204, 304],
    [521,  82, 582,   0, 898, 483, 271, 784, 826],
    [336, 866, 396, 898,   0, 482, 623, 259, 124],
    [236, 407, 101, 483, 482,   0, 263, 309, 383],
    [266, 234, 352, 271, 623, 263,   0, 589, 581],
    [311, 706, 204, 784, 259, 309, 589,   0, 166],
    [323, 765, 304, 826, 124, 383, 581, 166,   0]
    ]
#print(t)


# In[24]:


print("Shortest route for",N,"cities:")
d_max=0 # current distance
r=[0] # current shortest route
for i in range(1,N):
    d_max += d[i-1][i]
    r.append(i)
d_max += d[0][N-1]
r.append(0)
print("Current shortest route:",r)
print("Current shortest distance:",d_max)


# In[25]:


print("Optimize pairs")
d_old = d_max+1
while d_max < d_old:
    d_old = d_max
    for i in range(1,N-1):
        l1 = d[r[i-1]][r[i]] + d[r[i]][r[i+1]] + d[r[i+1]][r[i+2]]
        l2 = d[r[i-1]][r[i+1]] + d[r[i]][r[i+1]] + d[r[i]][r[i+2]]
        if l2 < l1:
            # swap r[i] and r[i+1]
            tmp = r[i]
            r[i] = r[i+1]
            r[i+1] = tmp
            d_max -= l1-l2
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[26]:


# Not needed?
print("Optimize triples")
d_old = d_max+1
while d_max < d_old:
    d_old = d_max
    for i in range(1,N-2):
        S = r[i-1]; # starting city, fixed
        A = r[i]; B = r[i+1]; C = r[i+2]; # next 3 cities
        F = r[i+3]; # finishing city, fixed
        l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        # one swap at a time is accounted in pairs above: ABC -> ACB, BAC
        l2 = d[S][B] + d[B][C] + d[C][A] + d[A][F] # BCA
        if l2 < l1:
            # swap ABC -> BCA
            r[i] = B; r[i+1] = C; r[i+2] = A;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][A] + d[A][B] + d[B][F] # CAB
        if l2 < l1:
            # swap ABC -> CAB
            r[i] = C; r[i+1] = A; r[i+2] = B;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][B] + d[B][A] + d[A][F] # CBA
        if l2 < l1:
            # swap ABC -> CBA
            r[i] = C; r[i+2] = A;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[27]:


# Not needed?
print("Optimize pairs")
d_old = d_max+1
while d_max < d_old:
    d_old = d_max
    for i in range(1,N-1):
        l1 = d[r[i-1]][r[i]] + d[r[i]][r[i+1]] + d[r[i+1]][r[i+2]]
        l2 = d[r[i-1]][r[i+1]] + d[r[i]][r[i+1]] + d[r[i]][r[i+2]]
        if l2 < l1:
            # swap r[i] and r[i+1]
            tmp = r[i]
            r[i] = r[i+1]
            r[i+1] = tmp
            d_max -= l1-l2
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[28]:


# Not needed?
print("Optimize triples")
d_old = d_max+1
while d_max < d_old:
    d_old = d_max
    for i in range(1,N-2):
        S = r[i-1]; # starting city, fixed
        A = r[i]; B = r[i+1]; C = r[i+2]; # next 3 cities
        F = r[i+3]; # finishing city, fixed
        l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        # one swap at a time is accounted in pairs above: ABC -> ACB, BAC
        l2 = d[S][B] + d[B][C] + d[C][A] + d[A][F] # BCA
        if l2 < l1:
            # swap ABC -> BCA
            r[i] = B; r[i+1] = C; r[i+2] = A;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][A] + d[A][B] + d[B][F] # CAB
        if l2 < l1:
            # swap ABC -> CAB
            r[i] = C; r[i+1] = A; r[i+2] = B;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][B] + d[B][A] + d[A][F] # CBA
        if l2 < l1:
            # swap ABC -> CBA
            r[i] = C; r[i+2] = A;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[29]:


print("Optimize 4 cities")
d_old = d_max+1
while d_max < d_old:
    d_old = d_max
    for i in range(1,N-3):
        S = r[i-1]; # starting city, fixed
        A = r[i]; B = r[i+1]; C = r[i+2]; D = r[i+3]; # next 4 cities
        F = r[i+4]; # finishing city, fixed
        l1 = d[S][A] + d[A][B] + d[B][C] + d[C][D] + d[D][F]
        # only need cyclic permutations
        l2 = d[S][B] + d[B][C] + d[C][D] + d[D][A] + d[A][F] # BCDA
        if l2 < l1:
            # swap ABCD -> BCDA
            r[i] = B; r[i+1] = C; r[i+2] = D; r[i+3] = A;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2]; D = r[i+3];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][D] + d[D][F]
        l2 = d[S][C] + d[C][D] + d[D][A] + d[A][B] + d[B][F] # CDAB
        if l2 < l1:
            # swap ABCD -> CDAB
            r[i] = C; r[i+1] = D; r[i+2] = A; r[i+3] = B;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2]; D = r[i+3];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][D] + d[D][F]
        l2 = d[S][D] + d[D][A] + d[A][B] + d[B][C] + d[C][F] # DABC
        if l2 < l1:
            # swap ABCD -> DABC
            r[i] = D; r[i+1] = A; r[i+2] = B; r[i+3] = C;
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2]; D = r[i+3];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][D] + d[D][F]
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# ## Results

# In[8]:


print("Current shortest route:\nStart\tEnd\tDist")
for i in range(len(r)-1):
    print(cityNames[r[i]],"\t",cityNames[r[i+1]],"\t",d[r[i]][r[i+1]])


# In[30]:


# Visualize the shortest route
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# customer locations, geographical coordinates in degrees
latitude = np.array([49.2347128, 48.4622135, 50.2678654, 47.8561438, 48.9117518, 50.401699, 48.5187443, 50.73977, 49.8326679])
longitude = np.array([28.3995942, 34.8602731, 28.6036778, 35.0352701, 24.6470892, 30.2525101, 32.1456232, 25.2639651, 23.9421958])
# convert to radians
Pi = np.pi
#latitude = Pi/180*latitude
#longitude = Pi/180*longitude
# https://en.wikipedia.org/wiki/Spherical_coordinate_system
theta = longitude # polar angle
phi = latitude-Pi/2 # azimuthal angle
R = 40000/(2*Pi) # radius of Earth 
import math
#X = R*np.sin(theta)*np.cos(phi) # x-coordinates in km
#Y = R*np.sin(theta)*np.sin(phi) # y-coordinates in km
#Z = R*np.cos(theta) # z-coordinates in km
X = 40000/360*longitude
Y = 40000/360*latitude

#,'Николаев','Одесса','Полтава','Ровно','Севастополь','Симферополь','Сумы','Тернополь','Ужгород','Харьков','Херсон','Хмельницкий','Черкассы','Чернигов','Черновцы'
# 'Донецк','Луганск',

def plot_tours(cityNames, r):
    tours = [[r[i], r[i+1]] for i in range(N)]
    plt.figure(1, figsize=(20,15))
    for s, tour in enumerate(tours):
        plt.plot([ X[tour[0]], X[tour[1]] ], [ Y[tour[0]], Y[tour[1]] ], color = "black", linewidth=0.5) # line
        plt.scatter(X[tour[1]], Y[tour[1]], marker = 'x', color = 'g', label = cityNames[tour[1]]) # dot
        plt.text(X[tour[1]]*1.001, Y[tour[1]]*1.001, cityNames[tour[1]], fontsize=12)
    #plt.scatter(0,0, marker = "o", color = 'b', label = "factory")
    plt.xlabel("X"), plt.ylabel("Y"), plt.title("Tours") #, plt.legend(loc = 1)
    plt.show()
    
plot_tours(cityNames, r)


# In[ ]:


# Distances from Google
import requests
import json
#Enter your source and destination city
originPoint = input("Please enter your origin city: ")
destinationPoint= input("Please enter your destination city: ")
#Place your google map API_KEY to a variable
apiKey = 'YOUR_API_KEY'
#Store google maps api url in a variable
url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
# call get method of request module and store respose object
r = requests.get(url + 'origins = ' + originPoint + '&destinations = ' + destinationPoint + '&key = ' + apiKey)
#Get json format result from the above response object
res = r.json()
#print the value of res
print(res)

