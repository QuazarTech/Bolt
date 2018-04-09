#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the module which contains the functions of the linear solver of Bolt.
It performs an FFT to map the given input onto the Fourier basis, and evolves 
each mode of the input independantly. It is to be noted that this module
can only be applied to systems with periodic boundary conditions.

NOTE: This module isn't parallelized to run across multiple devices/nodes.
"""

# In this code, we shall default to using the positionsExpanded form
# thoroughout. This means that the arrays defined in the system will
# be of the form:(N_p, N_s, N_q1, N_q2)

# Importing dependencies:
import numpy as np
import arrayfire as af
import socket
from petsc4py import PETSc
from inspect import signature

# Importing solver functions:
from .fields.fields import fields_solver
from .calculate_dfdp_background import calculate_dfdp_background
from .compute_moments import compute_moments as compute_moments_imported
from .file_io import dump, load

from bolt.lib.utils.bandwidth_test import bandwidth_test
from bolt.lib.utils.print_with_indent import indent
from bolt.lib.utils.broadcasted_primitive_operations import multiply
from bolt.lib.utils.fft_funcs import fft2, ifft2
from bolt.lib.utils.calculate_q import calculate_q_center
from bolt.lib.utils.calculate_p import calculate_p_center
from bolt.lib.utils.calculate_k import calculate_k

from . import timestep

class linear_solver(object):
    """
    An instance of this class' attributes contains methods which are used
    in evolving the system declared under physical system linearly. The 
    state of the system then may be determined from the attributes of the 
    system such as the distribution function and electromagnetic fields
    """

    def __init__(self, physical_system):
        """
        Constructor for the linear_solver object. It takes the physical
        system object as an argument and uses it in intialization and
        evolution of the system in consideration.

        Parameters
        ----------
        
        physical_system: The defined physical system object which holds
                         all the simulation information such as the initial
                         conditions, and the domain info is passed as an
                         argument in defining an instance of the linear_solver.
                         This system is then evolved, and monitored using the
                         various methods under the linear_solver class.
        """
        self.physical_system = physical_system

        # Storing Domain Information:
        self.q1_start, self.q1_end = physical_system.q1_start,\
                                     physical_system.q1_end
        self.q2_start, self.q2_end = physical_system.q2_start,\
                                     physical_system.q2_end
        self.p1_start, self.p1_end = physical_system.p1_start,\
                                     physical_system.p1_end
        self.p2_start, self.p2_end = physical_system.p2_start,\
                                     physical_system.p2_end
        self.p3_start, self.p3_end = physical_system.p3_start,\
                                     physical_system.p3_end

        # Getting Domain Resolution
        self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1
        self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2
        self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1
        self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2
        self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3
        
        # Getting number of species:
        N_s = self.N_species = len(physical_system.params.mass)

        if(type(physical_system.params.mass) == list):
            # Having a temporary copy of the lists to copy to af.Array:
            list_mass   = physical_system.params.mass.copy()
            list_charge = physical_system.params.charge.copy()

            # Initializing af.Arrays for mass and charge:
            # Having the mass and charge along axis 1:
            self.physical_system.params.mass   = af.constant(0, 1, N_s, dtype = af.Dtype.f64)
            self.physical_system.params.charge = af.constant(0, 1, N_s, dtype = af.Dtype.f64)

            for i in range(N_s):
                self.physical_system.params.mass[0, i]   = list_mass[i]
                self.physical_system.params.charge[0, i] = list_charge[i]

        # Initializing variable to hold time elapsed:
        self.time_elapsed = 0

        # Checking that periodic B.C's are utilized:
        if(    physical_system.boundary_conditions.in_q1_left   != 'periodic' 
           and physical_system.boundary_conditions.in_q1_right  != 'periodic'
           and physical_system.boundary_conditions.in_q2_bottom != 'periodic'
           and physical_system.boundary_conditions.in_q2_top    != 'periodic'
          ):
            raise Exception('Only systems with periodic boundary conditions \
                             can be solved using the linear solver'
                           )

        # Initializing DAs which will be used in file-writing:
        # This is done so that the output format used by the linear solver matches
        # with the output format of the nonlinear solver:
        self._da_dump_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof=(  self.N_species
                                                   * self.N_p1 
                                                   * self.N_p2 
                                                   * self.N_p3
                                                  )
                                             )

        # Getting the number of definitions in moments:
        attributes = [a for a in dir(self.physical_system.moments) if not a.startswith('_')]

        # Removing utility functions:
        if('integral_over_v' in attributes):
            attributes.remove('integral_over_v')

        self._da_dump_moments = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                    dof = self.N_species * len(attributes)
                                                   )

        # Printing backend details:
        PETSc.Sys.Print('\nBackend Details for Linear Solver:')
        PETSc.Sys.Print(indent('On Node: '+ socket.gethostname()))
        PETSc.Sys.Print(indent('Device Details:'))
        PETSc.Sys.Print(indent(af.info_str(), 2))
        # PETSc.Sys.Print(indent('Device Bandwidth = ' + str(bandwidth_test(100)) + ' GB / sec'))
        PETSc.Sys.Print()

        # Creating PETSc Vecs which are used in dumping to file:
        self._glob_f       = self._da_dump_f.createGlobalVec()
        self._glob_f_array = self._glob_f.getArray()

        self._glob_moments       = self._da_dump_moments.createGlobalVec()
        self._glob_moments_array = self._glob_moments.getArray()

        # Setting names for the objects which will then be
        # used as the key identifiers for the HDF5 files:
        PETSc.Object.setName(self._glob_f, 'distribution_function')
        PETSc.Object.setName(self._glob_moments, 'moments')

        # Intializing position, velocity and wave number arrays:
        self.q1_center, self.q2_center = \
            calculate_q_center(self.q1_start, self.q2_start,
                               self.N_q1, self.N_q2, 0,
                               self.dq1, self.dq2
                              )

        self.p1_center, self.p2_center, self.p3_center = \
            calculate_p_center(self.p1_start, self.p2_start, self.p3_start,
                               self.N_p1, self.N_p2, self.N_p3,
                               self.dp1, self.dp2, self.dp3, 
                              )
        
        # Converting dp1, dp2, dp3 to af.Array:
        self.dp1 = af.moddims(af.to_array(self.dp1), 1, self.N_species)
        self.dp2 = af.moddims(af.to_array(self.dp2), 1, self.N_species)
        self.dp3 = af.moddims(af.to_array(self.dp3), 1, self.N_species)

        self.k_q1, self.k_q2 = calculate_k(self.N_q1, self.N_q2,
                                           self.physical_system.dq1, 
                                           self.physical_system.dq2
                                          )

        # Assigning the function objects to methods of the solver:
        self._A_q    = self.physical_system.A_q
        self._A_p    = self.physical_system.A_p
        self._source = self.physical_system.source

        # Initializing f, f_hat and the other EM field quantities:
        self._initialize(physical_system.params)

    def get_dist_func(self):
        """
        Returns the distribution function in the same
        format as the nonlinear solver thereby allowing
        direct comparison with the distribution funcition
        of the nonlinear solver.
        """
        f = 0.5 * self.N_q2 * self.N_q1 * \
            af.real(ifft2(self.f_hat))

        return(f)

    def _calculate_k(self):
        """
        Initializes the wave numbers k_q1 and k_q2 which will be 
        used when solving in fourier space.
        """
        k_q1 = 2 * np.pi * np.fft.fftfreq(self.N_q1, self.dq1)
        k_q2 = 2 * np.pi * np.fft.fftfreq(self.N_q2, self.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q1 = af.to_array(k_q1)
        k_q2 = af.to_array(k_q2)

        k_q1 = af.reorder(k_q1, 2, 3, 0, 1)
        k_q2 = af.reorder(k_q2, 2, 3, 0, 1)

        af.eval(k_q1, k_q2)
        return(k_q1, k_q2)

    def _initialize(self, params):
        """
        Called when the solver object is declared. This function is
        used to initialize the distribution function, and the field
        quantities using the options as provided by the user. The
        quantities are then mapped to the fourier basis by taking FFTs.
        The independant modes are then evolved by using the linear
        solver.

        Parameters:
        -----------
        params: The parameters file/object that is originally declared 
                by the user.

        """
        # af.broadcast(function, *args) performs batched 
        # operations on function(*args):
        f = af.broadcast(self.physical_system.initial_conditions.\
                         initialize_f, self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center, params
                        )
        
        # Taking FFT:
        self.f_hat = fft2(f)

        # Since (k_q1, k_q2) = (0, 0) will give the background distribution:
        # The division by (self.N_q1 * self.N_q2) is performed since the FFT
        # at (0, 0) returns (amplitude * (self.N_q1 * self.N_q2))
        self.f_background = af.abs(self.f_hat[:, :, 0, 0])/ (self.N_q1 * self.N_q2)

        # Calculating derivatives of the background distribution function:
        self._calculate_dfdp_background()
   
        # Scaling Appropriately:
        # Except the case of (0, 0) the FFT returns
        # (0.5 * amplitude * (self.N_q1 * self.N_q2)):
        self.f_hat = 2 * self.f_hat / (self.N_q1 * self.N_q2) 

        rho_hat_initial = multiply(self.physical_system.params.charge,
                                   self.compute_moments('density', f_hat=self.f_hat)
                                  )
        
        self.fields_solver = fields_solver(self.physical_system,
                                           rho_hat_initial
                                          )

        return

    # Injection of solver methods from other files:
    # Assigning function that is used in computing the derivatives
    # of the background distribution function:
    _calculate_dfdp_background = calculate_dfdp_background

    # Time-steppers:
    RK2_timestep = timestep.RK2_step
    RK4_timestep = timestep.RK4_step
    RK5_timestep = timestep.RK5_step

    # Routine which is used in computing the 
    # moments of the distribution function:
    compute_moments = compute_moments_imported

    # Methods used in writing the data to dump-files:
    dump_distribution_function = dump.dump_distribution_function
    dump_moments               = dump.dump_moments

    # Used to read the data from file
    load_distribution_function = load.load_distribution_function
