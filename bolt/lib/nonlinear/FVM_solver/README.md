# FVM Solver:

This folder contains the routines that will be used when is it desired that the finite volume method method is to be used for solving. This folder contains the following:

- `reconstruction_methods/`: The folder contains the individual reconstruction method that can be user. Currently minmod, PPM and WENO5 have been implemented.

- `df_dt_fvm.py`: Returns the value of df_dt which has been evaluated for all the cells using FVM which is then passed to an integrator to get the value of the distribution function for the next timestep.

- `fvm_operator.py`: Since the nonlinear solver is capable of accepting different methods in q-space and p-space, operator splitting methods will need to be applied to maintain accuracy to the correct order. For this purpose we define an fvm_operator which will be fed appropriately to an operator splitting method.

- `reconstruct.py`: Contains the function which calls the appropriate reconstruction method as it has been defined under reconstruction_method_in_q and reconstruction_method_in_p as it has been defined under parameters.

- `riemann_solver.py`: Contains the Riemann solver functions which have been implemented(upwind_flux and lax_friedrichs_flux). It also contains a riemann_solver function which calls the appropriate Riemann solver depending upon the parameters riemann_solver_in_q and riemann_solver_in_p.

- `timestep_df_dt.py`: RK2 integrator which is used with df_dt. This function is defined independantly from the other integrators(in temporal_evolution/integrators) since the electrodynamic fields need to be evolved using the currents from the midpoint(at t + dt/2).
