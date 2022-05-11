# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:01:37 2021

@author: HERASAT-6
"""
import numpy as np

def reshaping(x):
    """
    Reshape the data created with numpy.
    """
    return np.reshape(x,(np.shape(x)[0], 1))

    "###############    Part 1: Temperature BC on walls    ##################"
def Temperature(NF, NFN, NFR, BCType, X, Y, IDS):
    u_BC = np.zeros(NF - NFN)    # Value of u on center of each boundary faces
    X_BC = np.zeros(NF - NFN)    # X-coordination of center of each boundary faces
    Y_BC = np.zeros(NF - NFN)    # Y-coordination of center of each boundary faces

    for i in range(NFN, NF):
    
        P1 = IDS[2][i] - 1
        P2 = IDS[3][i] - 1
        X_BC[i-NFN] = (X[P1] + X[P2]) / 2
        Y_BC[i-NFN] = (Y[P1] + Y[P2]) / 2

    # Boundary condition for 1st boundary(2nd region)    
    for i in range(0, NFR[1]):
        if BCType[1] == 2:
           u_BC[i] = 0 
    # Boundary condition for 2nd boundary(3rd region)          
    for i in range(NFR[1], NFR[1]+NFR[2]):
        if BCType[2] == 2:
           u_BC[i] = 0
    # Boundary condition for 3rd boundary(4th region)        
    for i in range(NFR[1]+NFR[2], NFR[1]+NFR[2]+NFR[3]):
        if BCType[3] == 2:
           u_BC[i] = 0 
    # Boundary condition for 4th boundary(5th region)        
    for i in range(NFR[1]+NFR[2]+NFR[3], NFR[1]+NFR[2]+NFR[3]+NFR[4]):
        if BCType[4] == 2:
           u_BC[i] = 1
           
    X_BC = reshaping(X_BC)
    Y_BC = reshaping(Y_BC)
    u_BC = reshaping(u_BC)

    return X_BC, Y_BC, u_BC