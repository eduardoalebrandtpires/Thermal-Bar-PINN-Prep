# Numerical Simulation of Heat Propagation in 1D and 2D Metal Bars
This project implements a numerical solution of the **1D and 2D heat equations** using the **finite difference method** in Python.
The goal is to simulate how heat propagates along a metal bar or a 2D plate under **idealized conditions**, and study the evolution of temperature over time until the system reaches thermal equilibrium.

## Objectives
* Solve the 1D and 2D heat equations numerically
* Study the stability of the explicit finite difference method
* Compare the numerical solution with the analytical steady-state solution
* Analyze numerical errors in the simulation
* Test convergence of the numerical method

## Physical Model
Heat propagation in the bar or plate is described by the heat equation:

- **1D:** dT/dt = α d²T/dx²  
- **2D:** dT/dt = α (d²T/dx² + d²T/dy²)

where:
* **T(x,t)** or **T(x,y,t)** is the temperature
* **α** is the thermal diffusivity of the material
* **x, y** are spatial coordinates
* **t** is time

All calculations are done under **idealized boundary and initial conditions**.
## Tools
* Python
* NumPy
* Matplotlib

## Future Work
This project can be extended by training a **Physics-Informed Neural Network (PINNs)** to solve the same heat equations and compare its performance with the traditional numerical solver.
