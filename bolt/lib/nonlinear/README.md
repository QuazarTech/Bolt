# Nonlinear Solver:

This is the root folder for the nonlinear solver module under which all the functions and methods of the nonlinear solver object have been defined. It consists of:

- `ASL_solver/`: This folder contains all the routines that will be called when the advective semi-lagrangian solver method is to be used.

- `fields/`: This folder contains the class definition for declaring a fields_solver, along with the methods which are used to evolve electrostatic/electrodynamic/user-defined fields.

- `file_io/`: This folder contains the FileIO routines which are used by the nonlinear solver. It contains definitions of functions which allow dumping the data to file, and loading data from file even when run in parallel.

- `FVM_solver/`: This folder contains all the routines that will be called when the finite volume method is to be used.

- `temporal_evolution/`: This folder contains the functions that are used in temporal evolution of the distribution function. It consists of operator splitting and time-integrators(RK methods) which are used when timestepping.

- `tests/`: This folder contains unit tests for the nonlinear solver.

- `utils/`: This folder contains the utility functions that are used in the nonlinear solver. These include functions which help in nicer formatting, bandwidth tests, etc...

- `apply_boundary_conditions.py`: This file contains the functions that are used to apply boundary conditions to the distribution function and the EM fields. The boundary conditions available are periodic, dirichlet, mirror, and shearing box boundary conditions.

- `communicate.py`: The functions are responsible for interzonal communication when the code is run in parallel. Additionally it also takes care of the application of periodic boundary conditions.

- `compute_moments.py`: This file contains the definition of the compute_moments function which returns the value of the moments as defined by the user under `src/`.

- `nonlinear_solver.py`: This file contains the class definition for creating the nonlinear solver object. This object acts as the interface through which the defined system is evolved.

- `timestep.py`: Contains the various timesplitting schemes with which the system can be evolved. It is to be noted that all methods under `timestep.py` are equivalent when considering FVM in q-space as well as p-space since there is no splitting involved.
