# Project Context

This project implements a distributed control algorithm for multi-agent systems using a two-time-scale dynamical system.

Agent dynamics:
x_dot_i = u_i

Each agent solves the local optimization problem

min 1/2 ||u_i||^2

subject to:
- inter-agent collision avoidance g_i^{i,j}
- obstacle avoidance g_obs^{i,k}
- connectivity constraint g_i^c
- CLF constraint g_i^{clf}
- actuator box constraint

The fast variables include:
- w_i ∈ R^2
- z_i^{i,j}, z_j^{i,j}
- z_i^c, z_j^c
- z_i^{clf}, z_j^{clf}
- y_i^{i,j}
- y_i^c
- y_i^{clf}

We assume obstacle and actuator dual variables remain zero.

Fast dynamics implemented so far:
- z_i^{i,j}
- z_i^c
- z_i^{clf}

The target region is defined by x_i^T Sigma x_i <= 1.
In code we use V_i(x_i) = x_i^T Sigma x_i - 1 + epsilon.

Next steps in development:
1. implement w_i dynamics
2. implement y dynamics
3. integrate local constraint builder
4. integrate the optimizer