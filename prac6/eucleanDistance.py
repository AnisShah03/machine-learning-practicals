from math import sqrt

def euclidean_distance(a,b):
    return sqrt(sum((e1-e2)**2 for e1,e2 in zip(a,b)))

row1 =[10,20,15,10,5]
row2 = [12,24,18,8,7]

dist= euclidean_distance(row1,row2)
print("Euclidean Distance :",dist)

##########################

from math import sqrt

def manhattan_distance(a,b):
    return sum(abs(e1-e2 )for e1, e2 in zip(a,b))

row1 =[10,20,15,10,5]
row2 = [12,24,18,8,7]

dist= manhattan_distance(row1,row2)
print("Manhattan Distance :",dist)











