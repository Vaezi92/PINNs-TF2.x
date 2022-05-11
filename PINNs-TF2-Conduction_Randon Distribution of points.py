
"##############################    Packages    ##############################"

import numpy as np

import time

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


"##############################    Dataset    ##############################"

def reshaping(x):
    """
    Reshape the data created with numpy. 
    """
    return x.reshape(x.shape[0], 1)

"########################    General information    #########################"

# Viscosity 
nu = 1e-02/np.pi

Nu = 100 # Number of points in the boundaries and initial value.  
Nf = 10000 # Number of collocated points.

SMALL = 1e-06

"#############################    Read Mesh    ##############################"

fileName = 'Mesh.txt'
fileObj = open(fileName)
key_value = {}


MeshDim = int()  # Dimension of Mesh
NP = int()       # Number of Points
NC = int()       # Number of Cells
NFa = int()       # Number of Faces
NR = int()       # Number of Regions
NFR = []         # Number of Faces of each Region
BCType = []      # Type of Boundary Condition 
BCTitle = []     # Region Name
FaceType = []    # Type of Faces
IDS = []         # 2D array with NR rows and 6 columns
Coordinates = [] # Coordinates Name(X,Y,Z)
X = []
Y = []
Z = []


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
       NFa = int(key_value[0])   
       
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
                   
    if C >= NR + 8 and C <= NR + NFa + 7:
       key_value = line.split()
       line = line.strip()
       FaceType.append(int(key_value[0]))        
       IDS.append([int(key_value[1]),int(key_value[2]),int(key_value[3]),int(key_value[4])])  


    if C == NR + NFa + 8:
       key_value = line.split()
       line = line.strip()
       Coordinates.append([str(key_value[1]),str(key_value[3]),str(key_value[5])]) 

    if C >= NR + NFa + 9 and C <= NR + NFa + NP + 8:
       key_value = line.split()
       line = line.strip()
       X.append(float(key_value[0]))        
       Y.append(float(key_value[1])) 
       Z.append(float(key_value[2])) 

"#########################    Boundary Condition    #########################"

# Boundary Condition
RanDist = np.random.uniform(0 + SMALL, 1, int(0.25*Nu))
RanDist = reshaping(RanDist)

# Boundary Condition at bottom wall
x_BC_1 = RanDist
y_BC_1 = 0*RanDist
u_BC_1 = 0*RanDist + 1

# Boundary Condition right wall
x_BC_2 = 0*RanDist + 1
y_BC_2 = RanDist
u_BC_2 = 0*RanDist

# Boundary Condition at top wall
x_BC_3 = RanDist
y_BC_3 = 0*RanDist + 1
u_BC_3 = 0*RanDist

# Boundary Condition at left wall
y_BC_4 = RanDist
x_BC_4 = 0*RanDist
u_BC_4 = 0*RanDist

# Boundary Condition
x_BC = np.concatenate((x_BC_1, x_BC_2, x_BC_3, x_BC_4), axis=0)
y_BC = np.concatenate((y_BC_1, y_BC_2, y_BC_3, y_BC_4), axis=0)
u_BC = np.concatenate((u_BC_1, u_BC_2, u_BC_3, u_BC_4), axis=0)


"#########################    Collocated Points    #########################"

x_collocated = np.random.uniform(0 + SMALL, 1 - SMALL, Nf)
x_collocated = reshaping(x_collocated)
y_collocated = np.random.uniform(0 + SMALL, 1 - SMALL, Nf)
y_collocated = reshaping(y_collocated)

xx, yy = np.concatenate((x_collocated, x_BC), axis=0), np.concatenate((y_collocated, y_BC), axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices((xx, yy))

batch_size = xx.shape[0]
train_dataset = train_dataset.batch(batch_size)

"#########################    Model Architecture    #########################"

n_targets = 1
n_hidden_layers, n_hidden_neurons = 2, 20

input_x = tf.keras.Input(shape=(1,))
input_y = tf.keras.Input(shape=(1,))
inputs = tf.keras.layers.concatenate([input_x, input_y])
initializer = tf.keras.initializers.GlorotNormal()
x = tf.keras.layers.Dense(n_hidden_neurons, activation='tanh', kernel_initializer=initializer, dtype=tf.float64)(inputs)
for hidden in range(n_hidden_layers-1):
    initializer = tf.keras.initializers.GlorotNormal()
    x = tf.keras.layers.Dense(n_hidden_neurons, activation='tanh', kernel_initializer=initializer, dtype=tf.float64)(x)
initializer = tf.keras.initializers.GlorotNormal()
outputs = tf.keras.layers.Dense(n_targets, activation='linear', kernel_initializer=initializer, dtype=tf.float64)(x)

model = tf.keras.Model(inputs=[input_x, input_y], outputs=outputs)

"##########################    Model Optimizer    ##########################"

# Instantiate an optimizer to train the model.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-02,
    decay_steps=100,
    decay_rate=0.96)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

"#########################    Model Loss Function    ########################"

loss_fn = tf.keras.losses.MeanSquaredError()

train_acc_metric = tf.keras.metrics.MeanAbsoluteError()
val_acc_metric = tf.keras.metrics.MeanAbsoluteError()

"#########################    Model Train Step    ###########################"

@tf.function
def train_step(x, y, x_BC, y_BC, u_BC):
    with tf.GradientTape() as tape:
        
        # Boundary conditions evaluation
        u_BC_pred = model([x_BC, y_BC], training=True)
        loss_BC = loss_fn(u_BC, u_BC_pred)
        
        # Collocated points evaluation - Differential Equation 
        with tf.GradientTape(persistent=True) as tape_diff:
            tape_diff.watch(x)
            tape_diff.watch(y)
            
            u_pred = model([x, y], training=True)
            
            u_x = tape_diff.gradient(u_pred, x)
            u_y = tape_diff.gradient(u_pred, y)
        u_xx = tape_diff.gradient(u_x, x)
        u_yy = tape_diff.gradient(u_y, y)
        
        del tape_diff
        
        e1 = u_xx + u_yy         
        loss_e1 = loss_fn(0, e1)
        
        # General loss function
        loss = loss_e1 + loss_BC
        
    # Regular backpropagation
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
    return loss, loss_e1, loss_BC

"#############################    Callbacks    ##############################"

def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    #No early stopping for 2*patience epochs 
    if len(LossList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False
    
"##############################    Training    ##############################"

file1 = open('lossfunction-epochs.Plt','w')    

epochs = 2000
start_time_outer = time.time()
loss_seq = []
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time_inner = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_train, loss_e1, loss_BC = train_step(x_batch_train, 
                                                  y_batch_train, 
                                                  x_BC, y_BC, u_BC)

        # Log every 200 batches.
        if step % 200 == 0:
            print(f"e1 loss (for one batch) at step {step}: {format(float(loss_e1), 'e')}")
            print(f"BC loss (for one batch) at step {step}: {format(float(loss_BC), 'e')}")
            print(f"Training loss (for one batch) at step {step}: {format(float(loss_train), 'e')}")
            print(f"Seen so far: {(step + 1) * batch_size} samples")
    
    # Stop criteria
    loss_seq.append(loss_train)  
    stopEarly = Callback_EarlyStopping(loss_seq, min_delta=1e-02, patience=1000)
    if stopEarly or loss_train < 1e-04:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
        print("Terminating training ")
        break
    file1.write(str(epoch)+"   " + str(float(loss_train))) 
    file1.write('\n')     
    
    print(f"Time taken: {round(time.time() - start_time_inner, 2)}s")
print(100*'=')
print(f"Total time taken: {round(time.time() - start_time_outer, 2)}s")
file1.close()


"###########################    Writing Results    ##########################"

file1 = open('Contour.Plt','w')
file1.write('Variables="x","y","u" \n')

# Boundary Condition Data  
for i in range(int(Nu)-1):
    file1.write(str(x_BC[i][0])+"   " + str(y_BC[i][0]) + "  " + str(u_BC[i][0])) 
    file1.write('\n')
    
# Collocation Points Data
for i in range(Nf+Nu):
    file1.write(str(xx[i][0])+"   " + str(yy[i][0]) + "  " + str(model.predict([xx[i], yy[i]])[0][0])) 
    file1.write('\n')
    
file1.close()
    



