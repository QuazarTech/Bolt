# Linear Solver:

This is the root folder for the linear solver module under which all the functions and methods of the linear solver object have been defined. It consists of:

- `fields/`: This folder contains the class definition for declaring a fields_solver object, along with the methods which are used to evolve electrostatic/electrodynamic/user-defined fields.

- `file_io/`: This folder contains the FileIO routines which are used by the linear solver. It contains definitions of functions which allow dumping the data to file, and loading data from file.

- `tests/`: This folder contains unit tests for the linear solver.

- `utils/`: This folder contains the utility functions that are used in the linear solver. These include functions which help in nicer formatting, bandwidth tests, etc...

- `calculate_dfdp_background.py`: This file contains the function that evaluates d(f_background)/dp using a fourth order stencil. This is needed to evaluate the contribution from fields on the evolution of the distribution function.

- `compute_moments.py`: This file contains the definition of the compute_moments function which returns the value of the moments as defined by the user under `src/`.

- `df_hat_dt.py`: This file returns df_hat/dt. This is then passed to an integrator to evolve fields_hat. It is to be noted that f_hat, and fields_hat have to be at the same temporal location since they are coupled. For this purpose, coupled integrators have been defined under `integrators.py` which are used when electrodynamic fields are to be evolved.

- `integrators.py`: Contains the various RK timestepping schemes which can be used to evolve a vector x using a slope function dx_dt, whose first argument is x. Additionally, it also contains RK timestepping schemes to evolve a coupled set of vectors(x, y) which accepts dx_dt, x, dy_dt, y as arguments, where dx_dt and dy_dt take their first arguments as x and y respectively.

- `linear_solver.py`: This file contains the class definition for creating the linear solver object. This object acts as the interface through which the defined system is evolved.

- `timestep.py`: Contains the various timestepping schemes with which the defined system may be evolved.
