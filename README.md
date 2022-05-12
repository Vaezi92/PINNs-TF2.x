# Physics Informed Neural Networks: a starting step for CFD specialists

 [Maziar Raisee et al. (2017)](https://maziarraissi.github.io/PINNs/) have developed Physics Informed Neural Networks approach to solving Partial Differential Equations. They present developments in the context of solving two main classes of problems: data-driven solutions and data-driven discovery of partial differential equations. In the direct approach (data-driven solution), PDE and boundary conditions of a problem are considered as an optimization problem and the goal is to decrease the loss function which is defined based on the objective function of the optimization problem. The main idea of the PINNs is to impose the temporal and spatial gradients in the architecture of a neural network based on chain rule differentiation and consider the PDE and boundary conditions as a loss function. You can see the non-dimensional governing equation of a natural convection problem and its architecture as bellow:
 
 ![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/NaturalConvection.png)
 ![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/NC-LossFunction.png)

## Pros & Cons

The PINNs approach is at its beginning step and has strong potential which will be explored in the future but there are main pros & cons at this time (June 2021):
- The PINNs is able to solve ill-posed problems.
- The PINNS is able to solve reverse problem (data-driven discovery).
- It has a weakness in predicting the boundary conditions (see 2D conduction or lid-driven cavity BCs).
- The cost of computations for training is high in comparison to CFD solutions (There are some predictions that the cost of computation may be decreased by the complexity of the problem in comparison to CFD solutions).
- There are no guarantees to reaching the best architecture in order to cover all problems. The best architecture for each problem can be determined using try and error or using the AutoML approaches.

## Notes for CFD specialist

![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2D%20Conduction%20-%20TemperatureField.png)
![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2D%20Conduction%20-%20TriMesh.png)
![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2DConduction-Lossfunction.png)





![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-Mesh.png)

![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-Pressurefield.png)

![alt text](https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-VelocityMagnitude.png)




## Prerequisites

- Python 3.x
- Numpy
- Tensorflow or Pytorch 1.0+

## Dataset

There are many public datasets for human activity recognition. You can refer to this survey article [Deep learning for sensor-based activity recognition: a survey](https://arxiv.org/abs/1707.03502) to find more.

In this demo, we will use UCI HAR dataset as an example. This dataset can be found in [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/).

Of course, this dataset needs further preprocessing before being put into the network. I've also provided a preprocessing version of the dataset as a `.npz` file so you can focus on the network (download [HERE](https://pan.baidu.com/s/1Nx7UcPqmXVQgNVZv4Ec1yg)). It is also highly recommended you download the dataset so that you can experience all the process on your own.

| #subject | #activity | Frequency |
| --- | --- | --- |
| 30 | 6 | 50 Hz |

## Usage

- For Pytorch (recommend), go to `pytorch` folder, config the folder of your data in `config.py', and then run `main_pytorch.py`.

- For tensorflow, run `main_tensorflow.py` file. The update of tensorflow version is stopped since I personally like Pytorch.

## Network structure

What is the most influential deep structure? CNN it is. So we'll use **CNN** in our demo. 

### CNN structure

Convolution + pooling + convolution + pooling + dense + dense + dense + output

That is: 2 convolutions, 2 poolings, and 3 fully connected layers. 

### About the inputs

That dataset contains 9 channels of the inputs: (acc_body, acc_total and acc_gyro) on x-y-z. So the input channel is 9.

Dataset providers have clipped the dataset using sliding window, so every 128 in `.txt` can be considered as an input. In real life, you need to first clipped the input using sliding window.

So in the end, we reformatted the inputs from 9 inputs files to 1 file, the shape of that file is `[n_sample,128,9]`, that is, every windows has 9 channels with each channel has length 128. When feeding it to Tensorflow, it has to be reshaped to `[n_sample,9,1,128]` as we expect there is 128 X 1 signals for every channel.

## Related projects

- [Must-read papers about deep learning based human activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/deep.md)
- [guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
- [aqibsaeed/Human-Activity-Recognition-using-CNN](https://github.com/aqibsaeed/Human-Activity-Recognition-using-CNN)
