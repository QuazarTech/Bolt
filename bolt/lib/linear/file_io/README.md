# FileIO Routines for the Linear Solver:

This folder contains the routines that will be used for fileIO. This folder contains the following files:

- `dump.py`: This file contains the routines for writing the simulation data to file. The routines make use of the PETSc viewer and output data in a HDF5 format. The routines included allow us to dump the distribution function, moments and the EM fields.

- `load.py`: This file contains the routines which are used to load the data from file to the solver object. These prove to be particularly useful when we need to restart the simulation from a particular time. The data is loaded back to the object using the `load_distribution_function()` and `load_EM_fields()` methods.
