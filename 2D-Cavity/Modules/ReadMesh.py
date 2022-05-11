# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:17:26 2021

@author: Mohammad Vaezi
"""
import numpy as np  
def ReadMesh():
                 
    fileName = 'Mesh.txt'
    fileObj = open(fileName)
    key_value = {}
    MeshDim = int()              # Dimension of Mesh
    NP = int()                   # Number of Points
    NC = int()                   # Number of Cells
    NF = int()                   # Number of Faces
    NR = int()                   # Number of Regions
    NFR = []                     # Number of Faces of each Region
    BCType = []                  # Type of Boundary Condition 
    BCTitle = []                 # Region Name
    FaceType = []                # Type of Faces
    IDS = []                     # Information of Grid Data Structure
    Coordinates = []             # Coordinates Name(X,Y,Z)
    X = []                       # X-coordination of eache point of the mesh
    Y = []                       # Y-coordination of eache point of the mesh
    Z = []                       # Z-coordination of eache point of the mesh
    SMALL = 1e-017                # A small number for division by zero
    C = 0

    for line in fileObj:
    
        C=C+1
    
        if C == 1:
           key_value = line.split()
           line = line.strip()
           MeshDim = int(key_value[0]) 
       
        if C == 2:
           key_value = line.split()
           line = line.strip()
           NP = int(key_value[0]) 
       
        if C == 3:
           key_value = line.split()
           line = line.strip()
           NC = int(key_value[0])   
       
        if C == 4:
           key_value = line.split()
           line = line.strip()
           NF = int(key_value[0])   
       
        if C == 5:
           key_value = line.split()
           line = line.strip()
           NR = int(key_value[0])   
       
        if C >= 7 and C <= NR + 6:
           key_value = line.split()
           line = line.strip()
           NFR.append(int(key_value[0]))
           BCType.append(int(key_value[1]))  
       
        if C == 7:
           key_value = line.split()
           line = line.strip()
           BCTitle.append(str(key_value[6])) 
                
        if C > 7 and C <= NR + 6:
           key_value = line.split()
           line = line.strip()
           BCTitle.append(str(key_value[5]))  
                   
        if C >= NR + 8 and C <= NR + NF + 7:
           key_value = line.split()
           line = line.strip()
           FaceType.append(int(key_value[0]))        
           IDS.append([int(key_value[1]),int(key_value[2]),int(key_value[3]),int(key_value[4])])  


        if C == NR + NF + 8:
           key_value = line.split()
           line = line.strip()
           Coordinates.append([str(key_value[1]),str(key_value[3]),str(key_value[5])]) 

        if C >= NR + NF + 9 and C <= NR + NF + NP + 8:
           key_value = line.split()
           line = line.strip()
           X.append(float(key_value[0]))        
           Y.append(float(key_value[1])) 
           Z.append(float(key_value[2])) 
           
    NFN = 0
    NFW = 0
    NFF = 0
    NFI = 0
    NFO = 0
    NFS = 0
    IDS = np.transpose(IDS) #2D array with 4 rows and NF columns

    for j in range(NR):
        if BCType[j] == 1:
           NFN = NFN + NFR[j]
        if BCType[j] == 2:
           NFW = NFW + NFR[j]
        if BCType[j] == 3:
           NFF = NFF + NFR[j]
        if BCType[j] == 4:
           NFI = NFI + NFR[j]
        if BCType[j] == 5:
           NFO = NFO + NFR[j]
        if BCType[j] == 6:
           NFS = NFS + NFR[j]

    NF1 = 0
    NF2 = NF1 + NFN

    A = np.zeros((NC,), dtype = float)             # Area of each Cell
    X_collocated = np.zeros((NC,), dtype = float)  # X-coordination of collocated points at center of cell
    Y_collocated = np.zeros((NC,), dtype = float)  # Y-coordination of collocated points at center of cell
    NX = np.zeros((NF,), dtype = float)            # X-component of normal vector on eache faces
    NY = np.zeros((NF,), dtype = float)            # X-component of normal vector on eache faces
    DA = np.zeros((NF,), dtype = float)            # Length of each faces
    DArea, SumX, SumY  = float(), float(), float()
    
    for i in range(NF1, NF2):
    
        ME = IDS[0][i] - 1
        NE = IDS[1][i] - 1
        P1 = IDS[2][i] - 1
        P2 = IDS[3][i] - 1
        DArea = X[P1] * Y[P2] - X[P2] * Y[P1]
        SumX = X[P1] + X[P2]
        SumY = Y[P1] + Y[P2]
        A[ME] = A[ME] + DArea
        X_collocated[ME] = X_collocated[ME] + SumX*DArea
        Y_collocated[ME] = Y_collocated[ME] + SumY*DArea
        A[NE] = A[NE] - DArea
        X_collocated[NE] = X_collocated[NE] - SumX*DArea
        Y_collocated[NE] = Y_collocated[NE] - SumY*DArea
         
    
    for i in range(NF2, NF):
    
        ME = IDS[0][i] - 1
        P1 = IDS[2][i] - 1 
        P2 = IDS[3][i] - 1

        DArea = X[P1] * Y[P2] - X[P2] * Y[P1]

        SumX = X[P1] + X[P2]
        SumY = Y[P1] + Y[P2]
        A[ME] = A[ME] + DArea
        X_collocated[ME] = X_collocated[ME] + SumX*DArea
        Y_collocated[ME] = Y_collocated[ME] + SumY*DArea

    for i in range(NC):
    # already I used SMALL, later will check why get warning:  RuntimeWarning: invalid
    # value encountered in double_scalars X_collocated[i] = X_collocated[i] / (6.0 * A[i] )
    # which leads to a nan!
        A[i]  = A[i] / 2.0
        X_collocated[i] = X_collocated[i] / (6.0 * A[i] + SMALL)
        Y_collocated[i] = Y_collocated[i] / (6.0 * A[i] + SMALL)

    
    for i in range(NF):
    
        P1 = IDS[2][i] - 1
        P2 = IDS[3][i] - 1
        NX[i] = Y[P2] - Y[P1]
        NY[i] = X[P1] - X[P2]
        DA[i] = np.sqrt(NX[i] * NX[i] + NY[i] * NY[i])
        
    "############     Writing Mesh     ###################"
    
    file1 = open('EBasedMesh.Plt','w')
    
    SFace = 0
    
    for i in range(NR):
        file1.write('TITLE = "Title" \n') 
        file1.write('VARIABLES  = X , Y , Z \n')
        file1.write('ZONE T= "' + str(BCTitle[i]) + '" ' + 'N= ' + str(NP) + ' ,' + 'E= ' + str(NFR[i]) + ' , ET=LINESEG, F=FEBLOCK' )
        file1.write('\n')
        for j in range(NP):
            file1.write(str(X[j]))
            file1.write('\n')
        for j in range(NP):
            file1.write(str(Y[j]))
            file1.write('\n')
        for j in range(NP):
            file1.write(str(Z[j]))
            file1.write('\n')
        
        for k in range(SFace , SFace + NFR[i]):
            file1.write(str(IDS[2][k]) + '      ' +str(IDS[3][k]))
            file1.write('\n')
        SFace = SFace + NFR[i] 
     
    file1.close() 
    
    return MeshDim, NP, NC, NF, NR, NFR, BCType, BCTitle, FaceType, IDS, Coordinates, X, Y, Z, NFN, NFW, NFF, NFI, NFO, NFS, X_collocated, Y_collocated, NX, NY, DA, A           