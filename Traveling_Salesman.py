#!/usr/bin/env python
# coding: utf-8

# In[40]:


# City names alphabetically
cityNames = ['Винница','Днепр','Житомир','Запорожье','Ивано-Франковск','Киев','Кропивницкий','Луцк','Львов']
#,'Николаев','Одесса','Полтава','Ровно','Севастополь','Симферополь','Сумы','Тернополь','Ужгород','Харьков','Херсон','Хмельницкий','Черкассы','Чернигов','Черновцы'
# 'Донецк','Луганск',
print(cityNames)
N = len(cityNames)
print("Number of cities: ",N)


# In[41]:


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


# In[42]:


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


# In[43]:


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


# In[44]:


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
            tmpA = r[i]
            r[i] = r[i+1] # B..
            tmpC = r[i+2]
            r[i+2] = tmpA # ..A
            r[i+1] = tmpC
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][A] + d[A][B] + d[B][F] # CAB
        if l2 < l1:
            # swap ABC -> CAB
            tmpA = r[i]
            r[i] = r[i+2] # C..
            tmpB = r[i+1]
            r[i+1] = tmpA # CA.
            r[i+2] = tmpB # CAB
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][B] + d[B][A] + d[A][F] # CBA
        if l2 < l1:
            # swap ABC -> CBA
            tmpA = r[i]
            r[i] = r[i+2] # C..
            r[i+2] = tmpA # CBA
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[45]:


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


# In[46]:


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
            tmpA = r[i]
            r[i] = r[i+1] # B..
            tmpC = r[i+2]
            r[i+2] = tmpA # ..A
            r[i+1] = tmpC
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][A] + d[A][B] + d[B][F] # CAB
        if l2 < l1:
            # swap ABC -> CAB
            tmpA = r[i]
            r[i] = r[i+2] # C..
            tmpB = r[i+1]
            r[i+1] = tmpA # CA.
            r[i+2] = tmpB # CAB
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
            l1 = d[S][A] + d[A][B] + d[B][C] + d[C][F]
        l2 = d[S][C] + d[C][B] + d[B][A] + d[A][F] # CBA
        if l2 < l1:
            # swap ABC -> CBA
            tmpA = r[i]
            r[i] = r[i+2] # C..
            r[i+2] = tmpA # CBA
            d_max -= l1-l2
            A = r[i]; B = r[i+1]; C = r[i+2];
    if d_max < d_old:
        print("Current shortest route:",r)
        print("Current shortest distance:",d_max)


# In[ ]:





# In[47]:


print("Current shortest route:\nStart\tEnd\tDist")
for i in range(len(r)-1):
    print(cityNames[r[i]],"\t",cityNames[r[i+1]],"\t",d[r[i]][r[i+1]])


# In[ ]:





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

