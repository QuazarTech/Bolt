*************************
Getting started with Bolt
*************************

Overview
---------

Bolt is organized such that a system is defined by making use of the ``physical_system`` class. The object created by ``physical_system`` is then passed as an argument to the solver objects.

Physical System
^^^^^^^^^^^^^^^
An instance of the ``physical_system`` object may be initialized by using::

    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             source,
                             moments
                            )

The arguments in the above command are all python modules/functions, where the details regarding the system being solved have been provided.

A detailed breakdown of what is to be contained in these files is demonstrated in the tutorials.

Solvers
^^^^^^^

The solver objects may be declared by using::

    nls = nonlinear_solver(system)
    ls  = linear_solver(system)

The physical system defined is then evolved using the various time-stepping methods available under each solver::

    for time_index, t0 in enumerate(time_array):
      print('Computing For Time =', t0)
      nls.strang_timestep(dt)
      ls.RK2_step(dt)

The abstracted information about the system may be obtained by using the ``compute_moments`` method available under each solver::

    density_nls = nls.compute_moments('density')
    density_ls  = ls.compute_moments('density')

The data about the evolved system can be dumped to file by making use of the methods ``dump_distribution_function``,``dump_moments`` and ``dump_EM_fields``

Running in Parallel
^^^^^^^^^^^^^^^^^^^

Bolt can be run in parallel across multiple nodes. To do so prefix the python command being executed with
``mpirun -n <nodes/devices>``. Make sure that ``num_devices`` is set correctly under ``params`` when running on nodes which contain more than a single accelerator(NOTE: The parallelization has only been implemented for the nonlinear solver. The linear solver can only take advantage of shared memory parallelism)

Tutorial Notebook
-----------------

`This <http://nbviewer.jupyter.org/github/ShyamSS-95/Bolt/blob/master/example_problems/nonrelativistic_boltzmann/quick_start/tutorial.ipynb>`_ notebook covers the basics of setting up and interacting with the primary features of the code. We consider the example problem of a 1D1V setup of the non-relativistic Boltzmann equation in which we observe the damping of the density with time. The same basic setup is also explored further after activating fields, and the source term(which is taken as the BGK collision operator for this example)  

Example Scripts
---------------

A wide range of examples are available under the ``example_problems`` subdirectory of the main code repository, which you can browse `here <https://github.com/QuazarTech/Bolt/tree/master/example_problems>`_.

These examples cover a wider range of use cases, including larger multidimensional problems designed for parallel execution. Most folders also have a ``README`` which gives context for the case that has been setup. Basic post-processing and plotting scripts are also provided with many problems.

These simulation and processing scripts may be useful as a starting point for implementing different problems and equation sets.
