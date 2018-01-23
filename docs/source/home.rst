****
Home
****

Overview
========

What is Bolt?
-------------

:math:`\texttt{Bolt}` is an open-source Python based framework for solving kinetic theory formulations uptil 5-dimensional phase space 
on a range of devices using the finite volume method, and/or the advective Semi-Lagrangian approach originally proposed by Cheng&Knorr. The framework is designed to solve a range of physical systems where the domain of interest can be mapped on to a rectangular grid. It is designed to target a range of hardware platforms via use of the `ArrayFire <http://arrayfire.com>`_ library, and is completely parallelized to run on large clusters by use of the `PETSc <https://www.mcs.anl.gov/petsc/>`_ library. 

The current release has the following capabilities:

- Dimensionality - Upto 2D3V phase space dimensionality
- Interpolation Methods - Linear, Cubic Spline
- Reconstruction Methods - minmod, PPM, WENO5
- Riemann Solvers - 1st Order Upwind flux, Local Lax-Friedrics flux
- Platforms - CPUs, OpenCL Devices, CUDA Devices
- Temporal Discretisation:
  
  - Time Splitting: Strang, Lie, SWSS
  
  - Time Stepping : Explicit - RK2, RK4, RK6
- Precision - Double
- Solution Files Exported - HDF5 (.h5, .hdf5)
