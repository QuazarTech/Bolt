# Bolt - A Solver Framework For Kinetic Theories:
[![Documentation Status](https://readthedocs.org/projects/qbolt/badge/?version=latest)](http://qbolt.readthedocs.io/en/latest/?badge=latest)[![Build Status](https://travis-ci.org/ShyamSS-95/Bolt.svg?branch=master)](https://travis-ci.org/ShyamSS-95/Bolt)

This framework provides methods for solving kinetic theory formulations uptil 5-dimensional phase space, and has been throughly unit tested. The framework consists of a linear as well as a non-linear solver. 

The non-linear solver can make use of the following solver methods:

* Conservative Finite Volume Method
    * Reconstruction Methods: minmod, PPM, WENO5
    * Riemann Solvers: Local Lax Friedrichs Flux, 1st order Upwind-Flux 
* Advective semi-Lagrangian solver based on the method proposed in [Cheng & Knorr, 1976](http://adsabs.harvard.edu/abs/1976JCoPh..22..330C). 

The framework has been written with ease of use and extensibility in mind, and can be used to obtain solution for any equation of the following form:

<p align="center"><img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/372221e63638d7fbbb468a0b9029d7a9.svg?invert_in_darkmode" align=middle width=450.2223pt height=36.953894999999996pt/></p>

Where <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/c612c13be517545496e8c6e2cb223153.svg?invert_in_darkmode" align=middle width=25.226190000000003pt height=22.381919999999983pt/>, <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/0f78a8e240089a351e31b7191713956c.svg?invert_in_darkmode" align=middle width=25.226190000000003pt height=22.381919999999983pt/>, <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/c01affe9953c41338ec92c5cb19e9dd5.svg?invert_in_darkmode" align=middle width=25.561965pt height=22.381919999999983pt/>, <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/60d7e869cdf701e26eaad5a4112ffff2.svg?invert_in_darkmode" align=middle width=25.561965pt height=22.381919999999983pt/>, <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/d38f278b9ebe01f0c37201b197e007d7.svg?invert_in_darkmode" align=middle width=25.561965pt height=22.381919999999983pt/>  and <img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/c90cd766eea8688c8b27fae1e3739f99.svg?invert_in_darkmode" align=middle width=30.926115000000003pt height=24.56552999999997pt/>  are terms that need to be coded in by the user.

The generalized structure that the framework uses can be found in `lib/`. All the functions have been provided docstrings which are indicative of their usage. Additionally, we have validated the solvers by solving the Boltzmann-Equation:

<p align="center"><img src="https://rawgit.com/ShyamSS-95/Bolt/master/.svgs/6012d33f73b29a6e67bdfd25286152d3.svg?invert_in_darkmode" align=middle width=766.6296pt height=38.464304999999996pt/></p>

The functions that have been used for solving the above equation may be referred to from `src/nonrelativistic_boltzmann`. The test problems we have solved can be found under `example_problems/nonrelativistic_boltzmann`. A README has been provided under each of the test folders for appropriate context.

## Dependencies:

The solver makes use of [ArrayFire](https://github.com/arrayfire/arrayfire) for shared memory parallelism, and [PETSc](https://bitbucket.org/petsc/petsc)(Built with HDF5 file writing support) for distributed memory parallelism and require those packages to be built and installed on the system of usage in addition to their python interfaces([arrayfire-python](https://github.com/arrayfire/arrayfire-python) and [petsc4py](https://bitbucket.org/petsc/petsc4py)). Additionally, following python libraries are also necessary:

* numpy
* h5py
* matplotlib
* pytest
* mpi4py

The documentation is built using sphinx, and requires the following dependencies to be built locally:

* sphinx
* sphinx_rtd_theme
* sphinx-autobuild
* numpydoc

## Authors

* **Shyam Sankaran** - [GitHub Profile](https://github.com/ShyamSS-95)
* **Mani Chandra** - [GitHub Profile](https://github.com/mchandra)