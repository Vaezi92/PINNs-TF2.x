# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:40:21 2021

@author: Mohammad Vaezi
"""
import numpy as np

def Contour(NP, NC, NF, NR, NFR, BCType, BCTitle, IDS, X, Y, Z, NFN, NFW, NFF, NFI, NFO, NFS, X_collocated, Y_collocated, u_collocated, X_BC, Y_BC, u_BC, NX, NY, DA, A, n_hidden_layers, n_hidden_neurons):

    Corn = EdgeToCell(NF, NC, IDS)
    
    file1 = open('Contour_'+str(NC)+'Cell_'+str(n_hidden_layers)+'layers_'+str(n_hidden_neurons)+'neurons'+'.Plt','w')
    file1.write('Variables="x","y","u" \n')
    file1.write('Zone N = '+str(NP)+'   E = '+str(NC)+'\n')
    file1.write('ZONETYPE=FEQUADRILATERAL DATAPACKING=BLOCK VARLOCATION=([3]=CELLCENTERED) \n')
#    file1.write('\n')
    for j in range(NP):
        file1.write(str(X[j])+'\n')
    for j in range(NP):
        file1.write(str(Y[j])+'\n')
#    for j in range(NP):
#        file1.write(str(Z[j])+'\n')
    for j in range(NC):
        file1.write(str(u_collocated[j][0])+'\n')
    for i in range(NC):
        P1 = Corn[0][i]
        P2 = Corn[1][i]
        P3 = Corn[2][i]
        P4 = Corn[3][i]
        if P4 == 0:
            P4 = P3
        file1.write(str(P1)+'   '+str(P2)+'    '+str(P3)+'    '+str(P4)+'\n')
        

    # Boundary Condition Data  
#    for i in range(NF-NFN):
#        file1.write(str(x_BC[i][0])+"   " + str(y_BC[i][0]) + "  " + str(u_BC[i][0])) 
#        file1.write('\n')
    
    # Collocation Points Data
#    for i in range(Nf+Nu):
#        file1.write(str(xx[i][0])+"   " + str(yy[i][0]) + "  " + str(u_collocated[0][0])) 
#        file1.write('\n')
#    file1.close()
    
def EdgeToCell(NF,NC,IDS):

    
    NCELL_EDGE = np.zeros((NC,), dtype = int)
    CELL_EDGE = np.zeros((4, NF), dtype = int)
    Corn = np.zeros((4,NF), dtype = int)
    
    for i in range(NF):
        ME = IDS[0][i] - 1
        NE = IDS[1][i] - 1
        if ME != -1:
           CELL_EDGE[NCELL_EDGE[ME]][ME] = i
           NCELL_EDGE[ME] += 1
            
        if NE != -1:
            CELL_EDGE[NCELL_EDGE[NE]][NE] = -i
            NCELL_EDGE[NE] += 1 
            
    for i in range(NC):
        for j1 in range(NCELL_EDGE[i] - 1):
            E1 = CELL_EDGE[j1][i] - 1
            if E1 >= 0:
                P2_E1 = IDS[3][E1]
            else:
                P2_E1 = IDS[2][-E1]
            for j2 in range(1, NCELL_EDGE[i]):
                E2 = CELL_EDGE[j2][i]
                if E2>=0:
                    P1_E2 = IDS[2][E2]
                else:
                    P1_E2 = IDS[3][-E2]
                if P2_E1 == P1_E2:
                    E = CELL_EDGE[j1 + 1][i]
                    CELL_EDGE[j1 + 1][i] = CELL_EDGE[j2][i]
            for i in range(NC):
                for j in range(NCELL_EDGE[i]):
                    E = CELL_EDGE[j][i]
                    if E >= 0:
                        P = IDS[2][E]
                    else:
                        P = IDS[3][-E]
                    Corn[j][i] = P
                
    return Corn