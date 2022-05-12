# Physics Informed Neural Networks: a starting step for CFD specialists

 [Maziar Raisee et al. (2017)](https://maziarraissi.github.io/PINNs/) have developed Physics Informed Neural Networks approach to solving Partial Differential Equations. They present developments in the context of solving two main classes of problems: data-driven solutions and data-driven discovery of partial differential equations. In the direct approach (data-driven solution), PDE and boundary conditions of a problem are considered as an optimization problem and the goal is to decrease the loss function which is defined based on the objective function of the optimization problem. The main idea of the PINNs is to impose the temporal and spatial gradients in the architecture of a neural network based on chain rule differentiation and consider the PDE and boundary conditions as a loss function. You can see the non-dimensional governing equation of a natural convection problem and its architecture as bellow:
 

| Governing equation | Architecture |
| --- | --- |
| <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/NC-LossFunction.png" width="400"> | <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/NaturalConvection.png" width="600"> |

## Two basic problem

- Two-dimensional conduction heat transfer

| Mesh | Loss function | Temperature field |
| --- | --- | ---|
| <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2D%20Conduction%20-%20TriMesh.png" width="400"> | <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2DConduction-Lossfunction.png" width="400"> | <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Conduction/Figs/2D%20Conduction%20-%20TemperatureField.png" width="400"> |

- Lid-driven cavity flow

| Mesh | Pressure field | Velocity Magnitude field |
| --- | --- | ---|
| <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-Mesh.png" width="400"> | <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-Pressurefield.png" width="400"> | <img src="https://github.com/Vaezi92/PINNs-TF2.x/blob/main/2D-Cavity/Figs/Cavity-VelocityMagnitude.png" width="400"> |


## Pros & Cons

The PINNs approach is at its beginning step and has strong potential which will be explored in the future but there are main pros & cons at this time (June 2021):
- The PINNs is able to solve ill-posed problems.
- The PINNS is able to solve reverse problem (data-driven discovery).
- It has a weakness in predicting the boundary conditions (see 2D conduction or lid-driven cavity BCs).
- The cost of computations for training is high in comparison to CFD solutions (There are some predictions that the cost of computation may be decreased by the complexity of the problem in comparison to CFD solutions).
- There are no guarantees to reaching the best architecture in order to cover all problems. The best architecture for each problem can be determined using try and error or using the AutoML approaches.

## Notes for CFD specialist
- According to the weaknesses of PINNs in the prediction of the field near the boundaries, It doesn't recommend to utilize this approach in the problems in which the quantities near the boundaries are important such as turbulence problems.
- Sample points distribution is random in the original version of PINNs which I think may cause some problems in the validation & verification process. Applying the specific density of the sample points in some zones of the field is a bit challenging. It is recommended to use traditional approaches of CFD to generate the mesh and consider the cell centers or nodes as sample points.
- The PINNs has more potential in data-driven discovery than direct solutions rather than traditional CFD approaches.
- 
## Prerequisites

- Python 3.9
- Numpy 1.20
- Tensorflow 2.8
