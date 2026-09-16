# Physics-Informed Neural Network for Fluid Flow Simulation

This project implements a Physics-Informed Neural Network (PINN) in MATLAB to model 
flows over periodic hills [2], using the Reynolds-Averaged Navier–Stokes (RANS) equations as the governing physics. 
Rather than learning purely from data, the network is also trained to satisfy the underlying 2-D physics, allowing it 
to generalize better with limited or noisy training data.

## Overview

- **Goal:** The goal of this project is to develop a PINN capable of predicting the flow 
            over a periodic hill without relying solely on labeled data. By incorporating 
            the RANS equations directly into the loss function, the network is constrained 
            toward physically consistent solutions, reducing dependence on large training 
            datasets and discouraging trivial, low-variance predictions (e.g. a near-constant 
            output that could minimize data loss without representing genuine flow physics).
- **Governing equations:** 2D steady incompressible RANS
- **Domain / geometry:**  Periodic hills of parameterized geometries [2]
- **Outputs predicted:** velocity field (u, v), pressure (p)

## Project Structure

| File | Description |
|---|---|
| `PHLL_PINN.m` | Main script - Loads training data, initialises the neural network and trains the PINN and exports trainedPINN.mat |
| `modelLoss.m` | Overall model loss function called within PHLL_PINN.m |
| `wallLoss.m` | Wall Boundary Condition loss function called within modelLoss.m |
| `dataLoss.m` | DNS data loss function called within modelLoss.m |
| `physicsResidualLoss.m` | Physics informed loss function within modelLoss.m |
| `convertArray.mat` | Used to convert arrays into dlarrays, whilst also checking if a GPU is available, so it can use a gpuArray instead for faster running |
| `fourierFeatures.mat` | Used to map the input coordiantes into high frequency signals to minimize spectral bias [1] |
| `fcRWFLayer.mat` | Custom neural network layer that incorporates random weight factorizations (RWF) to improve performance [1] |
| `buildNetwrok.mat` | Function to build the neural network using given hyperparameters |
| `filtered_data.mat` | Filtered DNS data [3] which is used to train the PINN. |
| `plotModelVsData.m` | Ploting script of the PINN vs DNS data to produce the figures given below |
| `trainedPINN.mat` | Given trained PINN from which the data presented below is extracted from using plotModelvsData.m |
| `PINN_Figures` | Folder containing .png and .fig of the figures seen below. PHLL_PINN.m stores the training loss figure and plotModelvsData.m stores the comparison plots here. |


## How to Run

1. Clone or download this repository.
2. Open MATLAB (tested on version [R2025a], requires Deep Learning Toolbox).
3. Ensure `filtered_data.mat` is in the same folder as the scripts.
4. Run the training script:
   ```matlab
   PHLL_PINN.m
   ```
   This trains the network and saves the result to `trainedPINN.mat`.
5. Run the visualization/evaluation script:
   ```matlab
   plotModelVsData.m
   ```
   Note: `trainedPINN.mat` already contains a fully trained model ready to run `plotModelVsData.m`

   This loads the trained model and generates the comparison plots shown below. The figures are stored in PINN_Figures as .png and.fig files.

## Methodology

The physics model is embedded into the training of the PINN as a loss function. 
There are three loss terms: the physics residual, the data residual (against the DNS data), 
and the enforced wall boundary condition residual. The DNS Reynolds stresses were interpolated 
to compute the spatial derivatives required for the physics residual, while automatic differentiation 
was used to compute the velocity and pressure derivatives. The Adams optimizer was used to train the model, 
with 5000 epochs found to be sufficient to produce reliable results. Techniques such as random weight 
factorization and Fourier feature mapping [1] were used to enhance the model.

## Results

**Predicted vs. True $u^*$ Velocity Field**

![Result 1](PINN_Figures\contour_u.png)


**Predicted vs. True $v^*$ Velocity Field**

![Result 2](PINN_Figures\contour_v.png)


**Predicted vs. True $p^*$ Pressure Distribution**

![Result 3](PINN_Figures\contour_p.png)


**Predicted vs. True $u^*$ Streamwise Distribution**

![Result 3](PINN_Figures\profile_u.png)


**Predicted vs. True Error Plots**

![Result 3](PINN_Figures\scatter_comparison.png)


## Requirements

- MATLAB R2025a
- Deep Learning Toolbox

## Future Work

- Addition of temporal weights [1] to improve the performance of the training


## Acknowledgments

Submitted as part of the [MATLAB & Simulink Challenge Project Hub] project:
*Fluid Flow Simulation Using Physics-Informed Neural Networks*. <br>
Available: https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Fluid%20Flow%20Simulation%20Using%20Physics-Informed%20Neural%20Networks

[1] S. Wang, S. Sankaran, H. Wang, and P. Perdikaris, <br> 
"An Expert's Guide to Training Physics-Informed Neural Networks," [Online]. <br>
Available: https://arxiv.org/abs/2308.08468

[2] H. Xiao, J.-L. Wu, S. Laizet, and L. Duan, <br>
"Flows over periodic hills of parameterized geometries: A dataset for data-driven turbulence modeling from direct simulations," 
Computers & Fluids, vol. 200, 2020. <br>
doi: 10.1016/j.compfluid.2020.104431

[3] R. McConkey, "Turbulence Modelling Using Machine Learning" <br>
[Dataset], Kaggle, version 3. [Online]. <br>
Available: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset

