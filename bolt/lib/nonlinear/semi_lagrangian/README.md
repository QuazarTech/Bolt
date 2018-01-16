# ASL Solver:

This folder contains the routines that will be used when is it desired that the advective semi-Lagrangian method is to be used for solving. The user has the option of choosing to use the solver method used in p-space and q-space between 'ASL' and 'FVM'. This folder contains the following files:

- `asl_operators.py`: This file contains the routines for, advection in q-space, advection in p-space and solving for the source term. The appropriate routines are called depending on the parameters method_in_q_space and method_in_p_space which are defined by the user.

- `interpolation_routines.py`: This contains the function that finds the origin of the characteristics and interpolates at the location. Contains the interpolation routines f_interp_2d which performs the interpolation in q-space and f_interp_p_3d which performs the interpolation in p-space.
