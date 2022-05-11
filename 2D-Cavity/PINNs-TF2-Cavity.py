
"##############################    Packages    ##############################"
import sys
import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


"#########################    Reshaping function    #########################"

def reshaping(x):
    """
    Reshape the data created with numpy.
    """
    return np.reshape(x,(np.shape(x)[0], 1))

"#############################    Read Mesh    ##############################"
"""
Produce meshes using Ansys mesh and export mesh as ICEM input. Open the meshes 
using ICEM and save the meshes for Ansys fluent with the name of fluent.msh. Copy 
the fluent.msh to 'fluent to mesh' file and run the fortran code(Main_ReadMshFile)
to produce Mesh.txt. Open the Mesh.txt file and change the BC values as bellow:
    Boundary Condition:    BC values:
    Interior               1   
    Wall                   2  
    Riemann far field      3
    Velocity inlet         4       
    Pressure outlet        5
    Symmetry               6
        
"""   
sys.path.append('I:\litterature Review\PINNs-TF2-Cavity\Modules')
import ReadMesh as RM

MeshDim, NP, NC, NF, NR, NFR, BCType, BCTitle, FaceType, IDS, Coordinates, X, Y, Z, NFN, NFW, NFF, NFI, NFO, NFS, X_collocated, Y_collocated, NX, NY, DA, A = RM.ReadMesh()

NF1 = 0
NF2 = NF1 + NFN

NFW1 = NF2
NFW2 = NFW1+NFW

NFF1 = NFW2
NFF2 = NFF1 + NFF

NFI1 = NFF2
NFI2 = NFI1 + NFI

NFO1 = NFI2
NFO2 = NFO1 + NFO

NFS1 = NFO2
NFS2 = NFS1 + NFS





for n_hidden_layers in [4]:
    for n_hidden_neurons in [15]:
        
        "#########################    Boundary Condition    #########################"
        sys.path.append('I:\litterature Review\PINNs-TF2-Cavity\Modules')
        import BoundaryCondition as BC
        X_BC, Y_BC, u_BC, v_BC, p_BC = BC.Wall(NF, NFN, NFR, BCType, X, Y, IDS)

        "#########################    Collocated Points    #########################"
        X_collocated = reshaping(X_collocated)
        Y_collocated = reshaping(Y_collocated)   
        xx, yy = np.concatenate((X_collocated, X_BC), axis=0), np.concatenate((Y_collocated, Y_BC), axis=0)       
        train_dataset = tf.data.Dataset.from_tensor_slices((xx, yy))
        
        "#########################    Model Architecture    #########################"
        batch_size = xx.shape[0]
        train_dataset = train_dataset.batch(batch_size)
        sys.path.append('I:\litterature Review\PINNs-TF2-Cavity\Modules')
        import PhysicsInformedNeuralNetwork as PINN
        u_collocated, u_BC, v_collocated, v_BC, p_collocated, p_BC = PINN.PINN(batch_size, train_dataset, X_collocated, Y_collocated, X_BC, Y_BC, u_BC, v_BC, p_BC, NC, n_hidden_layers, n_hidden_neurons)
        
        "###########################    Writing Results    ##########################"
        sys.path.append('I:\litterature Review\PINNs-TF2-Cavity\Modules')
        import WritingResults as WR
        WR.Contour(NP, NC, NF, NR, NFR, BCType, BCTitle, IDS, X, Y, Z, NFN, NFW, NFF, NFI, NFO, NFS, X_collocated, Y_collocated, u_collocated, v_collocated, p_collocated, X_BC, Y_BC, u_BC, NX, NY, DA, A, n_hidden_layers, n_hidden_neurons)

        

    



