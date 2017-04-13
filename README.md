# Boltzmann-Equation Solver Package

The solver package consists of a Cheng-Knorr solver as well as a linear theory solver. The results obtained from both methods are verified against each other. The solvers have been written in terms of library functions which can be called from a separate python file or more conveniently from an iPython notebook. In both the solver libraries a common configuration file is used to setup the initial conditions of the simulation along with the simulation run-time parameters. 

## Getting Started:

Clone the repo to your local machine, and add the folder to your python path. These steps may be performed by:

```bash
git clone https://github.com/ShyamSS-95/Boltzmann-Solver.git
cd Boltzmann-Solver
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Dependencies:

The following python libraries are prerequisites to run the solver package:

* numpy
* arrayfire-python
* h5py
* matplotlib
* scipy

## Usage:

The test files written under each of the solver libraries are indicative of the structure that is to be used in every simulation run. Additionally each of the solver functions have their own docstring that briefly describe the usage of the function along with the parameters read and the output generated.

## Authors

* **Shyam Sankaran** - [GitHub Profile](https://github.com/ShyamSS-95)
* **Mani Chandra** - [GitHub Profile](https://github.com/mchandra)

## In development:

* Automated unit test framework(Currently manual visual checks are necessary)
* Parallelization of the Cheng-Knorr code to run on multiple nodes.(mainly applicable to 2D + 2V)
* Addition of other collision operators(currently only BGK operator has been implemented)