**********
User Guide
**********

Getting Started
===============

Downloading the Source:
-----------------------

Bolt can be obtained `here <https://github.com/QuazarTech/Bolt>`_.

Dependencies:
-------------

Overview:
^^^^^^^^^

Bolt has a hard dependency on Python 3+ and the following
Python packages:

1. `mpi4py <http://mpi4py.scipy.org/>`_ 
2. `numpy <http://www.numpy.org/>`_ 
3. `h5py <http://www.h5py.org/>`_ 
4. `pytest <https://pypi.python.org/pypi/pytest>`_
5. `scipy <https://www.scipy.org/>`_
6. `matplotlib <https://http://matplotlib.org/>`_
7. `petsc4py <https://bitbucket.org/petsc/petsc4py>`_ 
8. `arrayfire <https://github.com/arrayfire/arrayfire-python>`_ 

Before installing the above python packages, the following libraries need to be installed
so that their python wrappers can function: 

Building ArrayFire:
^^^^^^^^^^^^^^^^^^^

- Clone the `arrayfire <https://github.com/arrayfire/arrayfire>`_ repository
- Build using the instructions that have been provided `here <https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux>`_ 

Building PETSc:
^^^^^^^^^^^^^^^

- Clone the `petsc <https://bitbucket.org/petsc/petsc>`_ repository
- Build for production(without debugging) or developement(debugging enabled) following the instructions `here <http://www.mcs.anl.gov/petsc/documentation/installation.html>`_

Installation:
-------------

Before running Bolt it is first necessary to either install
the software using the provided ``setup.py`` installer(TODO) or add 
the root directory to ``PYTHONPATH`` using::

    user@computer ~/Bolt$ export PYTHONPATH=.:$PYTHONPATH

Once the build of ArrayFire and PETSc is completed install the python dependencies
using::

    user@computer ~/Bolt$ pip install -r requirements.txt

Running Bolt
============

Overview:
---------

Bolt is organized such that a system is defined by making use of the 
``physical_system`` class. The object created by ``physical_system`` is then
passed as an argument to the solver objects.

Physical System:
----------------
An instance of the ``physical_system`` object may be initialized by using::

    system = physical_system(domain,\
                             boundary_conditions,\
                             params,\
                             initialize,\
                             advection_terms,\
                             sink_or_source,\
                             moment_defs
                            )

The arguments in the above command are all python files, where the definitions have been provided.
A detailed breakdown of what is to be contained is provided below:(TODO)(For now refer to ``example_problems/nonrelativistic_boltzmann/testing_folder``)

Solvers:
--------

The solver objects may be declared by using::

    nls = nonlinear_solver(system)
    ls  = linear_solver(system)

The physical system defined is then evolved using the various time-stepping methods 
available under each solver::

    for time_index, t0 in enumerate(time_array):
      print('Computing For Time =', t0)
      nls.strang_timestep(dt)
      ls.RK2_step(dt)

The abstracted information about the system may be obtained by using the ``compute_moments`` method available under each solver::

    density_nls = nls.compute_moments('density')
    density_ls  = ls.compute_moments('density')

The data about the evolved system can be dumped to file by making use of the methods ``dump_distribution_function`` and ``dump_variables``

Running in Parallel
^^^^^^^^^^^^^^^^^^^

Bolt can be run in parallel across multiple node. To do so prefix the python command being executed with
``mpirun -n <nodes/devices>``.(NOTE: The parallelization has only been implemented for
the nonlinear solver. The linear solver can only take advantage of shared memory parallelism)