# Fields Solver:

This folder contains the routines that will be used when we need to couple a fields solver with the evolution of the nonlinear solver. The fields solver routines are completely self-contained and only require the source terms such as charge density and currents to be inputed from the nonlinear solver. This folder contains the following:

- `fields_solver.py`: The file that contains the class which will be used to initialize a fields_solver object. The attributes of this class are initialized depending upon the input of the user, and are then evolved by the methods of the object.

- `electrostatic_solvers/`: The folder contains all the electrostatic solvers that the fields_solver object can make use of. Currently, the solvers available are:

    - FFT Solver: Solves the Poisson Equation using FFTs. Can only be used in serial and with periodic boundary conditions.

    - SNES Solver: The Scalable Nonlinear Equations Solvers (SNES) component of PETSc is used to solve the Poisson equation. This is a much more versatile solver capable of making use of several solver methods in addition to preconditioners. Additionally this solver can be run in parallel.

- `electrodynamic_solvers/`: The folder contains all the electrodynamic solvers that the fields_solver object can make use of. Currently, only the explicit FDTD solver has been implemented.
