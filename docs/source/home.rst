****
Home
****

Overview
========

What is NameTBD?
-------------
NameTBD is an open-source Python based framework for solving
advection type problems with sources/sinks uptil 5-dimensional phase space 
on a range of devices using the Semi-Lagrangian approach of Cheng&Knorr. The 
framework is designed to solve a range of physical systems where the domain of 
interest can be mapped on to a rectangular grid. It is designed to target a range
of hardware platforms via use of the ArrayFire library, and is completely 
parallelized to run on large clusters by use of the PETSc library. 
The current release has the following capabilities:

- Dimensionality - Upto 2D3V phase space dimensionality
- Platforms - CPUs, OpenCL Devices, CUDA Devices
- Temporal Discretisation:

  - Time Splitting: Strang, Lie

  - Time Stepping : Explicit - RK2, RK4, RK6
- Precision - Double