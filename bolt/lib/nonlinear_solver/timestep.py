#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

# Importing functions used used for time-splitting and time-stepping:
from .temporal_evolution import operator_splitting_methods as split
from .temporal_evolution import integrators

# Importing solver functions:
from .FVM_solver.df_dt_fvm import df_dt_fvm
from .FVM_solver.timestep_df_dt import fvm_timestep_RK2

from .interpolation_routines import f_interp_2d
from .EM_fields_solver.fields_step import fields_step

# Defining the operators:
# When using FVM:
def op_fvm_q(self, dt):

    # Solving for tau = 0 systems
    if(af.any_true(self.physical_system.params.tau(self.q1_center, self.q2_center,
                                                   self.p1, self.p2, self.p3
                                                  ) == 0
                  )
      ):
        if(self.performance_test_flag == True):
            tic = af.time()

        self.f = self._source(self.f, self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.compute_moments, 
                              self.physical_system.params, 
                              True
                             ) 

        fvm_timestep_RK2(self, dt)
        
        if(self.performance_test_flag == True):
            af.sync()
            toc = af.time()
            self.time_sourcets += toc - tic

        af.eval(self.f)
        return
    

    if(self.performance_test_flag == True):
        tic = af.time()

    fvm_timestep_RK2(self, dt)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    

    af.eval(self.f)
    return

# When using advective SL method:
# Advection in q-space:
def op_advect_q(self, dt):
    self._communicate_f()
    self._apply_bcs_f()
    f_interp_2d(self, dt)

    return

# Used to solve the source term:
# df/dt = source
def op_solve_src(self, dt):
    if(self.performance_test_flag == True):
        tic = af.time()

    # Solving for tau = 0 systems
    if(af.any_true(self.physical_system.params.tau(self.q1_center, self.q2_center,
                                                   self.p1, self.p2, self.p3
                                                  ) == 0
                  )
      ):
        self.f = self._source(self.f, self.q1_center, self.q2_center,
                              self.p1, self.p2, self.p3, 
                              self.compute_moments, 
                              self.physical_system.params, 
                              True
                             ) 

    else:
        self.f = integrators.RK2(self._source, self.f, dt,
                                 self.q1_center, self.q2_center,
                                 self.p1, self.p2, self.p3, 
                                 self.compute_moments, 
                                 self.physical_system.params
                                )

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic
    
    return

# Used to solve the Maxwell's equations and
# perform the advections in p-space:
def op_fields(self, dt):
    return(fields_step(self, dt))

def check_divergence(self):
    if(   af.any_true(af.isinf(self.f))
       or af.any_true(af.isnan(self.f))
      ):
        raise SystemExit('Solver Diverging!')

def lie_step(self, dt):
    """
    Advances the system using a lie-split 
    scheme. This scheme is 1st order accurate in
    time.

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    self.dt            = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.charge_electron != 0
          ):
            split.strang(self, op_fvm_q, op_fields, dt)

        else:
            op_fvm_q(self, dt)


    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.charge_electron == 0):
            split.lie(self, op_advect_q, op_solve_src, dt)

        else:
            def op_advect_q_and_solve_src(self, dt):
                return(split.lie(self, 
                                 op1 = op_advect_q,
                                 op2 = op_solve_src, 
                                 dt = dt
                                )
                      )

            split.lie(self, op_advect_q_and_solve_src, op_fields, dt)

    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return

def strang_step(self, dt):
    """
    Advances the system using a strang-split 
    scheme. This scheme is 2nd order accurate in
    time.

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    self.dt            = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.charge_electron != 0
          ):
            split.strang(self, op_fvm_q, op_fields, dt)

        else:
            op_fvm_q(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.charge_electron == 0):
            split.strang(self, op_advect_q, op_solve_src, dt)

        else:
            def op_advect_q_and_solve_src(self, dt):
                return(split.strang(self, 
                                    op1 = op_advect_q,
                                    op2 = op_solve_src, 
                                    dt = dt
                                   )
                      )

            split.strang(self, op_advect_q_and_solve_src, op_fields, dt)
    
    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return

def swss_step(self, dt):
    """
    Advances the system using a SWSS-split 
    scheme. This scheme is 2nd order accurate in
    time.

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    self.dt            = dt
    
    if(self.performance_test_flag == True):
        tic = af.time()
    
    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.charge_electron != 0
          ):
            split.strang(self, op_fvm_q, op_fields, dt)

        else:
            op_fvm_q(self, dt)


    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.charge_electron == 0):
            split.swss(self, op_advect_q, op_solve_src, dt)

        else:
            def op_advect_q_and_solve_src(self, dt):
                return(split.swss(self, 
                                  op1 = op_advect_q,
                                  op2 = op_solve_src, 
                                  dt = dt
                                 )
                      )

            split.swss(self, op_advect_q_and_solve_src, op_fields, dt)

    check_divergence(self)
    self.time_elapsed += dt 
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return

def jia_step(self, dt):
    """
    Advances the system using the Jia split scheme.

    NOTE: This scheme is computationally expensive, and 
          should only be used for testing/debugging

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    self.dt            = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.charge_electron != 0
          ):
            split.strang(self, op_fvm_q, op_fields, dt)

        else:
            op_fvm_q(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.charge_electron == 0):
            split.jia(self, op_advect_q, op_solve_src, dt)

        else:
            def op_advect_q_and_solve_src(self, dt):
                return(split.jia(self, 
                                 op1 = op_advect_q,
                                 op2 = op_solve_src, 
                                 dt = dt
                                )
                      )

            split.jia(self, op_advect_q_and_solve_src, op_fields, dt)
    
    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return
