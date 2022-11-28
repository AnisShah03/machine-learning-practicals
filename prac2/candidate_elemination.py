from operator import ge, le
import re
import pandas as pd
import numpy as np

data = pd.DataFrame(data=pd.read_csv("prac2\data.csv"))

print(data)
concepts = np.array(data.iloc[:,0:-1])

target = np.array(data.iloc[:,-1])

print("\n",target)

print("\n",concepts)

def learn(concepts, tareget):
    speacific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")

    print("\n specific_h:",speacific_h)
    general_h=[["?" for i in range(len(speacific_h))] for i in range(len(speacific_h))]

    print("\ngeneral_h:",general_h)
    print("\nconcepts:",concepts)

    for i ,h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(speacific_h)):

                #print("h[x]",h[x])

                if h[x] != speacific_h[x]:
                    speacific_h[x] =='?'
                    general_h[x][x]=='?'
            
            for x in range(len(speacific_h)):

                if h[x] != speacific_h[x]:
                    general_h[x][x]== speacific_h[x]
                else:
                    general_h[x][x]=="?"

        print("\n steps of candidate elimination algorithm :" ,i+1)
        print("\n Specific_h:",i+1)
        print(speacific_h,"\n")
        print("general_h:",i+1)
        print(general_h)

    indices = [i for i, val in enumerate (general_h) if val ==['?','?','?','?','?','?']]

    print("\n Indices",indices)

    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    return speacific_h,general_h
s_final,g_final = learn(concepts,target)

print("\n Final Specific_h",s_final,sep="\n")
print("\n Final General_h",g_final,sep="\n")






            

