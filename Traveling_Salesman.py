# City names alphabetically
cityNames = ['Винница','Днепр','Житомир','Запорожье','Ивано-Франковск','Киев','Кропивницкий','Луцк']
#,'Львов','Николаев','Одесса','Полтава','Ровно','Севастополь','Симферополь','Сумы','Тернополь','Ужгород','Харьков','Херсон','Хмельницкий','Черкассы','Чернигов','Черновцы'
# 'Донецк','Луганск',
print(cityNames)
N = len(cityNames)
print("Number of cities: ",N)

# Distance matrix using maps.google.com. Last included city is 'Луцк'
d = [
    [  0, 588, 128, 651, 368, 268, 322, 389],
    [588,   0, 596,85.4, 962, 480, 247, 875],
    [128, 596,   0, 690, 413, 140, 392, 261],
    [651,85.4, 690,   0,1026, 538, 310, 952],
    [368, 962, 413,1026,   0, 560, 694, 263],
    [268, 480, 140, 538, 560,   0, 303, 400],
    [322, 247, 392, 310, 694, 303,   0, 670],
    [389, 875, 261, 952, 263, 400, 670,   0]
    ]
#print(d)

# Time by car
t = [
    [  0, 484,  99, 521, 336, 236, 266, 311],
    [484,   0, 474,  82, 866, 407, 234, 706],
    [ 99, 474,   0, 582, 396, 101, 352, 204],
    [521,  82, 582,   0, 898, 483, 271, 784],
    [336, 866, 396, 898,   0, 482, 623, 259],
    [236, 407, 101, 483, 482,   0, 263, 309],
    [266, 234, 352, 271, 623, 263,   0, 589],
    [311, 706, 204, 784, 259, 309, 589,   0]
    ]
#print(t)

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
 
print("Current shortest route:\nStart\tEnd\tDist")
for i in range(len(r)-1):
    print(cityNames[r[i]],"\t",cityNames[r[i+1]],"\t",d[r[i]][r[i+1]])