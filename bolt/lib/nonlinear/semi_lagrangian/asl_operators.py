#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file defines methods which will be used when the 
advective semilagrangian method is to be used.

The operations of solving for the q-space, collisions and
p-space are defined independantly so that they maybe called 
appropriately by the time-splitting operator methods. 
"""

import arrayfire as af
from ..temporal_evolution import integrators
from .interpolation_routines import f_interp_2d, f_interp_p_3d
from bolt.lib.utils.broadcasted_primitive_operations import multiply

# Advection in q-space:
def op_advect_q(self, dt):
    """
    Solves the following part of the equations specified:
    
    df/dt + A_q1 df/dq1 + A_q2 df/dq2 = 0

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self._communicate_f()
    self._apply_bcs_f()
    f_interp_2d(self, dt)

    af.eval(self.f)
    return

# Used to solve the source term:
def op_solve_src(self, dt):
    """
    Evolves the source term of the equations specified:
    
    df/dt = source
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """

    if(self.performance_test_flag == True):
        tic = af.time()

    # Solving for tau = 0 systems:
    if(self.physical_system.params.instantaneous_collisions == True):
        self.f = self._source(self.f, self.time_elapsed,
                              self.q1_center, self.q2_center,
                              self.p1_center, self.p2_center, self.p3_center, 
                              self.compute_moments, 
                              self.physical_system.params, 
                              True
                             )

    if(    self.physical_system.params.source_enabled == True 
       and self.physical_system.params.instantaneous_collisions != True
      ):
        self.f = integrators.RK2(self._source, self.f, dt, 
                                 self.time_elapsed, 
                                 self.q1_center, self.q2_center,
                                 self.p1_center, self.p2_center, 
                                 self.p3_center, self.compute_moments, 
                                 self.physical_system.params
                                )
    
    af.eval(self.f)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic
    
    return

def op_fields(self, dt):
    """
    Evolves the following part of the equations specified:

    df/dt + A_p1 df/dp1 + A_p2 df/dp1 + A_p3 df/dp1 = 0
    
    Parameters
    ----------
    dt : double
         Time-step size to evolve the system
    """

    if(self.performance_test_flag == True):
        tic = af.time()
    
    if(self.physical_system.params.fields_solver == 'electrostatic'):
        rho = multiply(self.physical_system.params.charge,
                       self.compute_moments('density')
                      )
        self.fields_solver.compute_electrostatic_fields(rho)
    
    # Evolving fields:
    if(self.physical_system.params.fields_solver == 'fdtd'):
        
        J1 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v1_bulk')
                     )  # (i + 1/2, j + 1/2)
        J2 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v2_bulk')
                     )  # (i + 1/2, j + 1/2)
        J3 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v3_bulk')
                     )  # (i + 1/2, j + 1/2)

        self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

    f_interp_p_3d(self, dt)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldstep += toc - tic
    
    return
