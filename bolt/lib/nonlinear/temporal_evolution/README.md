# Temporal Evolution:

This folder contains the routines which are used for temporal evolution. This involves integrators and operator splitting methods. This folder contains the following files:

- `integrators.py`: This file includes all RK based integrators which can be used in evolving the source term(ie. op_solve_src). It includes methods to evolve any system that returns dx_dt which takes x as it's first argument using RK2, RK4 and RK5 methods.

- `operator_splitting_methods.py`: This file includes the operator splitting methods when using any two operators op1, op2 and takes the timestep which will be passed to the individual operators. Currently Lie, Strang, SWSS and Jia methods of operator splitting have been implemented.
