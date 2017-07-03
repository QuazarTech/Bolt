# SLaPy - A Python-based Semi-Lagrangian Solver Framework:

This framework provides methods for solving an advection equation with sources/sinks uptil 5-dimensional phase space. The framework consists of a linear as well as a non-linear solver. The non-linear solver is a semi-Lagrangian solver based on the method proposed in [Cheng & Knorr, 1976](http://adsabs.harvard.edu/abs/1976JCoPh..22..330C). The framework has been written with ease of use and extensibility in mind, and can be used to obtain solution for any equation of the following form:

<p align="center"><img alt="\begin{align*}&#10;\frac{\partial f}{\partial t} + T_1 \frac{\partial f}{\partial q_1} + T_2 \frac{\partial f}{\partial q_2} + T_3 \frac{\partial f}{\partial q_3} + T_4 \frac{\partial f}{\partial q_4} + T_5 \frac{\partial f}{\partial q_5} = T_6&#10;\end{align*}" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/None/.svgs/382b10006744bebc4c3584f60d42e784.svg?invert_in_darkmode" align=middle width="385.83104999999995pt" height="36.953894999999996pt"/></p>

Where <img alt="$T_1$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/None/.svgs/b1aadae6dafc7da339f61626db58e355.svg?invert_in_darkmode" align=middle width="16.098390000000002pt" height="22.381919999999983pt"/> to <img alt="$T_6$" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/None/.svgs/7a05d4075902c41f2194c3e72cc65519.svg?invert_in_darkmode" align=middle width="16.098390000000002pt" height="22.381919999999983pt"/> are terms that need to be coded in by the user.

The generalized structure that the framework uses can be found in `lib/`. All the functions have been provided docstrings which are indicative of their usage. Additionally, we have validated the solvers by solving the Boltzmann-Equation:

<p align="center"><img alt="\begin{align*}&#10;\frac{\partial f}{\partial t} + v_x \frac{\partial f}{\partial x} + v_y \frac{\partial f}{\partial y} + q(\vec{E} + \vec{v} \times \vec{B})_x \frac{\partial f}{\partial v_x} + q(\vec{E} + \vec{v} \times \vec{B})_y \frac{\partial f}{\partial v_y} + q(\vec{E} + \vec{v} \times \vec{B})_z \frac{\partial f}{\partial v_z} = C[f] = -\frac{f - f_0}{\tau}&#10;\end{align*}" src="https://rawgit.com/ShyamSS-95/Boltzmann_solver/None/.svgs/fb517d5e70e914e4e30148467dd0b571.svg?invert_in_darkmode" align=middle width="735.1954499999999pt" height="38.464304999999996pt"/></p>

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