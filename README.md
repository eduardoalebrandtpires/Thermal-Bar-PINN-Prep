# Numerical Simulation of Heat Propagation in a 1D Iron Bar
This project implements a numerical solution of the **1D heat equation** using the **finite difference method** in Python.
The goal is to simulate how heat propagates along a metal bar and study the evolution of temperature over time until the system reaches thermal equilibrium.

## Objectives
* Solve the 1D heat equation numerically
* Study the stability of the explicit finite difference method
* Compare the numerical solution with the analytical steady-state solution
* Analyze numerical errors in the simulation

## Physical Model
Heat propagation in the bar is described by the heat equation:

dT/dt = α d²T/dx²

where:
* **T(x,t)** is the temperature
* **α** is the thermal diffusivity of the material
* **x** is the spatial coordinate
* **t** is time

## Tools
* Python
* NumPy
* Matplotlib

## Future Work
This project will be extended by training a **Physics-Informed Neural Network (PINNs)** to solve the same heat equation and compare its performance with the traditional numerical solver.
