import numpy as np
mangow= np.array([1,1,1])
orangew= np.array([1,1,1])
ma= 10
alpha=0.00001
o=25
m=15
Jm=0
mango=[56,81,119,22,103,57,80,118,21,104,57,82,118,20,102]
orange=[70,101,133,37,119,69,102,132,38,118,69,100,134,38,120]
TRHarr=np.array([[73,91,87,102,69,74,91,88,101,68,73,92,87,103,68],[67,88,134,43,96,66,87,134,44,96,66,87,135,43,97],[43,64,58,37,70,43,65,59,37,71,44,64,57,36,70]])
MANarr=np.array([56,81,119,22,103,57,80,118,21,104,57,82,118,20,102])
ORAarr=np.array([70,101,133,37,119,69,102,132,38,118,69,100,134,38,120])
#print(TRHarr[0])
fM=np.dot(mangow,TRHarr)+ma
dert=0.25
delJ = 10
delJo=10
J_final = 0
Jo_final = 0

#print(fM)
fO=np.dot(orangew,TRHarr)+o
while(abs(delJ) > 0.001):
    derr=0
    derm=0
    derh=0
    Jm = 0
    dert = 0
    fM=np.dot(mangow,TRHarr)+ma
    for i in range(m):
        dert= dert+1/m * (TRHarr[0][i]*(fM[i]-mango[i]))
        derr=derr+1/m * (TRHarr[1][i]*(fM[i]-mango[i]))
        derh=derh+1/m * (TRHarr[2][i]*(fM[i]-mango[i]))
        derm=derm+1/m * (fM[i]-mango[i])
        Jm=Jm+(1/m*1/2)*(fM[i]-mango[i])*(fM[i]-mango[i])
    der=np.array([dert,derr,derh])
    J_init = Jm
    delJ = J_init - J_final
    J_final = Jm
    

    mangow= mangow - alpha*der
    ma= ma - alpha*derm
    #mangow=tmpm
    #ma=tmpma
    #print(Jm)
print("mango=")
print(mangow)
print(ma)
    
    
while(abs(delJo) > 0.001):
    deror=0
    derom=0
    deroh=0
    Jo = 0
    derot = 0
    fO=np.dot(orangew,TRHarr)+o
    for i in range(m):
        derot= derot+1/m * (TRHarr[0][i]*(fO[i]-ORAarr[i]))
        deror=deror+1/m * (TRHarr[1][i]*(fO[i]-ORAarr[i]))
        deroh=deroh+1/m * (TRHarr[2][i]*(fO[i]-ORAarr[i]))
        derom=derom+1/m * (fO[i]-ORAarr[i])
        Jo=Jo+(1/m*1/2)*(fO[i]-ORAarr[i])*(fO[i]-ORAarr[i])

    dero=np.array([derot,deror,deroh])
    Jo_init = Jo
    delJo = Jo_init - Jo_final
    Jo_final = Jo
    

    orangew= orangew - alpha*dero
    o= o - alpha*derom
    #mangow=tmpm
    #ma=tmpma
    #print(Jo)
print("orange=")
print(orangew)
print(o)




 