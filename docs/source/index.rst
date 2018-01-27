.. Bolt documentation master file, created by
   sphinx-quickstart on Fri Aug  4 17:22:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: underline
    :class: underline

Welcome to Bolt's Documentation!
********************************

About Bolt:
===========

:math:`\texttt{Bolt}` is a flexible framework for solving kinetic theory formulations, making use of the finite volume and/or advective semi-lagrangian method. Additionally, it also consists of a linear solver which is primarily used in verifying the results given by the nonlinear solver. The code is open-source and developed by the research division at `Quazar Technologies <http://quazartech.com>`_, Delhi where it used to study device physics and astrophysical plasmas

The code is written in Python and features an easy-to-use interface, where the user provides input through a ``physical system`` object which holds details about the system solved. The ``physical_system`` object is declared by defining the advection terms, and the source term for the system of interest. Additionally, the physical simulation also requires details such as the domain information, and initial conditions.

:math:`\texttt{Bolt}` is capable of running on CPUs and GPUs, and has been parallelized to be able to run efficiently across several nodes/devices.

Doc Contents
============
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   home
   theory
   installation
   getting_started
   units

Quick-Reference:
----------------

.. autosummary::
   bolt.lib.physical_system
   bolt.lib.linear.linear_solver
   bolt.lib.nonlinear.nonlinear_solver


.. toctree::
    :maxdepth: 2

Other Links
===========

Learn more about Bolt by visiting the

* Code repository: http://github.com/QuazarTech/Bolt
* Documentation: http://qbolt.rtfd.io
