# A Python-based Semi-Lagrangian Solver Framework:

This framework provides methods for solving an advection equation with sources/sinks uptil 5-dimensional phase space. The framework consists of a linear as well as a non-linear solver. The non-linear solver is a semi-Lagrangian solver based on the method proposed in [Cheng & Knorr, 1976](http://adsabs.harvard.edu/abs/1976JCoPh..22..330C). The framework has been written with ease of use and extensibility in mind, and can be used to obtain solution for any equation of the following form:

<p align="center"><img alt="\begin{align*}&#10;\frac{\partial f}{\partial t} + A_q1 \frac{\partial f}{\partial q_1} + A_q2 \frac{\partial f}{\partial q_2} + A_p1 \frac{\partial f}{\partial p_1} + A_p2 \frac{\partial f}{\partial p_2} + A_p3 \frac{\partial f}{\partial p_3} = g(f)&#10;\end{align*}" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/6f9d83a1f781580576b02ea3ca73df5a.svg?invert_in_darkmode" align=middle width="458.40299999999996pt" height="36.953894999999996pt"/></p>

Where <img alt="$A_q1$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/8b6326999a89ab4f1c4a07ceccf130fb.svg?invert_in_darkmode" align=middle width="27.731055000000005pt" height="22.381919999999983pt"/>, <img alt="$A_q2$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/af735a9aac328732e413f63fd2ba0123.svg?invert_in_darkmode" align=middle width="27.731055000000005pt" height="22.381919999999983pt"/>, <img alt="$A_p1$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/90823cf3e090021a62e898abc6821a84.svg?invert_in_darkmode" align=middle width="28.069634999999998pt" height="22.381919999999983pt"/>, <img alt="$A_p2$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/bd93eda5ed937799b2dd026ab5438ac5.svg?invert_in_darkmode" align=middle width="28.069634999999998pt" height="22.381919999999983pt"/>, <img alt="$A_p3$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/d933e7665c83bc73a480d40c069b3b09.svg?invert_in_darkmode" align=middle width="28.069634999999998pt" height="22.381919999999983pt"/>  and <img alt="$g(f)$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/c90cd766eea8688c8b27fae1e3739f99.svg?invert_in_darkmode" align=middle width="30.926115000000003pt" height="24.56552999999997pt"/>  are terms that need to be coded in by the user.

The generalized structure that the framework uses can be found in `lib/`. All the functions have been provided docstrings which are indicative of their usage. Additionally, we have validated the solvers by solving the Boltzmann-Equation:

<p align="center"><img alt="\begin{align*}&#10;\frac{\partial f}{\partial t} + v_x \frac{\partial f}{\partial x} + v_y \frac{\partial f}{\partial y} + \frac{q}{m}(\vec{E} + \vec{v} \times \vec{B})_x \frac{\partial f}{\partial v_x} + \frac{q}{m}(\vec{E} + \vec{v} \times \vec{B})_y \frac{\partial f}{\partial v_y} + \frac{q}{m}(\vec{E} + \vec{v} \times \vec{B})_z \frac{\partial f}{\partial v_z} = C[f] = -\frac{f - f_0}{\tau}&#10;\end{align*}" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/refactor_library/.svgs/6012d33f73b29a6e67bdfd25286152d3.svg?invert_in_darkmode" align=middle width="766.6296pt" height="38.464304999999996pt"/></p>

`src/` contains the relevant files which were used to make the framework solve for the Boltzmann-Equation.

## Dependencies:

The solver makes use of [ArrayFire](https://github.com/arrayfire/arrayfire) for shared memory parallelism, and [PETSc](https://bitbucket.org/petsc/petsc)(Built with hdf5 file writing support) for distributed memory parallelism and require those packages to be built and installed on the system of usage in addition to their python interfaces([arrayfire-python](https://github.com/arrayfire/arrayfire-python) and [petsc4py](https://bitbucket.org/petsc/petsc4py)). Additionally, following python libraries are also necessary:

* numpy
* h5py(used in file writing/reading)
* matplotlib(used in postprocessing the data-generated)
* pytest

## Authors

* **Shyam Sankaran** - [GitHub Profile](https://github.com/ShyamSS-95)
* **Mani Chandra** - [GitHub Profile](https://github.com/mchandra)