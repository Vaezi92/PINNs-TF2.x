# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 08:40:47 2021

@author: Mohammad Vaezi
"""

import numpy as np
import time
import tensorflow as tf

def PINN(batch_size, train_dataset, X_collocated, Y_collocated, X_BC, Y_BC, u_BC, NC, n_hidden_layers, n_hidden_neurons):
    
    "########    Part 1: Define Architecture and Initialization    ##########"
    n_targets = 1
    #n_hidden_layers, n_hidden_neurons = 1, 10

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

    "####################    Part 2: Model Optimizer    #####################"
    # Instantiate an optimizer to train the model.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-02,
        decay_steps=100,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    "###################    Part 3: Model Loss Function    ##################"
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    #train_acc_metric = tf.keras.metrics.MeanAbsoluteError()
    #val_acc_metric = tf.keras.metrics.MeanAbsoluteError()

    "###################    Part 4: Model Train Step    #####################"

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

    "#######################    Part 5: Callbacks    ########################"

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
    
    "########################    part 6: Training    ########################"

    file1 = open('lossfunction-epochs'+str(NC)+'Cell_'+str(n_hidden_layers)+'layers_'+str(n_hidden_neurons)+'neurons'+'.Plt','w')    

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
                                                      X_BC, Y_BC, u_BC)

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
    u_collocated = model.predict([X_collocated, Y_collocated])
    u_BC = model.predict([X_BC, Y_BC])
    return u_collocated, u_BC
