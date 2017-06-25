# Boltzmann-Equation Solver Package

The solver package consists of a Cheng-Knorr solver as well as a linear theory solver. The results obtained from both methods are verified against each other. The solvers have been written in terms of library functions which can be called from a separate python file(refer to `run_ck.py` and `run_lt.py` under the `run_folder/`). In both the solver libraries a common configuration file is used to setup the initial conditions and boundary conditions of the simulation along with the simulation run-time parameters.

## Getting Started:

Clone the repo to your local machine, and add the folder to your python path. These steps may be performed by:

```bash
git clone https://github.com/ShyamSS-95/Boltzmann-Solver.git
cd Boltzmann-Solver
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Dependencies:

The solver makes use of [ArrayFire](https://github.com/arrayfire/arrayfire) for shared memory parallelism, and [PETSc](https://bitbucket.org/petsc/petsc)(Built with hdf5 file writing support) for distributed memory parallelism and require those packages to be built and installed on the system of usage in addition to their python interfaces([arrayfire-python](https://github.com/arrayfire/arrayfire-python) and [petsc4py](https://bitbucket.org/petsc/petsc4py)). Additionally, following python libraries are also necessary:

* numpy
* h5py(used in file writing/reading)
* matplotlib(used in postprocessing the data-generated)
* pytest

## Usage:

To get started, look at `run_ck.py` and `run_lt.py` under `run_folder/`. Changes need to be made to `params.py` to indicate the parameters of the system which you intend to evolve for. Commenting has been added appropriately in each of these files to indicate the purpose of each function call. Additionally, docstrings have been provided for all the functions(not all - few in development), which can be viewed easily by typing `function_name?` from an IPython shell. One of the standard case we have been considering thusfar is the evolution of a system with a small density perturbation. Try making changes to the `params.py`, and see the plot for density amplitudes as given by the linear theory code, and the Cheng-Knorr:
```bash
ipython run_lt.py
ipython run_ck.py
ipython plot_density_amplitudes.py
```

In addition to the various unit tests, automated convergence tests that check the implementation of the physics have been added. Convergence tests for all the cases can be performed by running `run_all.sh`. Alternatively, to test a particular case, navigate to the appropriate folder, and execute the following set of commands:

```bash
ipython run_lt.py
ipython run_ck.py
py.test
```

## Authors

* **Shyam Sankaran** - [GitHub Profile](https://github.com/ShyamSS-95)
* **Mani Chandra** - [GitHub Profile](https://github.com/mchandra)

## In development:

* Addition of other collision operators(currently only BGK operator has been implemented)
* Additional unit checks for the Cheng-Knorr solvers 
* Unit tests for the linear theory code
* Evolving for multiple species
* Implementation of a conservative and positivity preserving algorithm.
* More thorough documentation