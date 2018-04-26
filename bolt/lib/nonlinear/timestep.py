#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Since our solver allows the capability to use different methods in
p-space and q-space, we need to use operator splitting methods to maintain 
the correct spatio-temporal order of accuracy.For this purpose, we have defined
independant steps as operators which may be passed appropriately to the operator 
splitting methods. This file contains the functions lie_step, strang_step, swss_step
and jia_step which call the corresponding operator splitting methods according to the
parameters defined by the user.

NOTE: When FVM is used in the q-space as well as the p-space, there is no splitting introduced
      since the operation in q-space and p-space are carried out in a single step. In such a 
      case, all the methods are equivalent.
""" 

import arrayfire as af
import numpy as np

# Importing functions used used for time-splitting and time-stepping:
from .temporal_evolution import operator_splitting_methods as split

# Importing solver functions:
from .finite_volume.fvm_operator import op_fvm
from .semi_lagrangian.asl_operators import op_advect_q, op_solve_src, op_fields

def check_divergence(self):
    """
    Used to terminate the program if a blowup occurs in any segment
    of the solver, resulting in the values becoming infinity or 
    undefined.
    """

    if(   af.any_true(af.isinf(self.f))
       or af.any_true(af.isnan(self.f))
      ):
        raise SystemExit('Solver Diverging!')

def lie_step(self, dt):
    """
    Advances the system using a lie-split scheme. 
    This scheme is 1st order accurate in time.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self.dt = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.fields_enabled == True
          ):
            split.lie(self, op_fvm, op_fields, dt)

        else:
            op_fvm(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.fields_enabled == True):
            
            def op_advect_q_and_solve_src(self, dt):
                
                return(split.lie(self, 
                                 op1 = op_advect_q,
                                 op2 = op_solve_src, 
                                 dt = dt
                                )
                      )

            if(self.physical_system.params.solver_method_in_p == 'ASL'):
                split.lie(self, op_advect_q_and_solve_src, op_fields, dt)

            # For FVM in p-space:
            else: 
                split.lie(self, op_advect_q_and_solve_src, op_fvm, dt)

        else:
            split.lie(self, op_advect_q, op_solve_src, dt)

    af.eval(self.f)
    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return

def strang_step(self, dt):
    """
    Advances the system using a strang-split scheme. This scheme is 
    2nd order accurate in time.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self.dt = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.fields_enabled == True
          ):
            split.strang(self, op_fvm, op_fields, dt)

        else:
            op_fvm(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.fields_enabled == True):
            
            def op_advect_q_and_solve_src(self, dt):
                
                return(split.strang(self, 
                                    op1 = op_advect_q,
                                    op2 = op_solve_src, 
                                    dt = dt
                                   )
                      )

            if(self.physical_system.params.solver_method_in_p == 'ASL'):
                split.strang(self, op_advect_q_and_solve_src, op_fields, dt)

            # For FVM in p-space:
            else: 
                split.strang(self, op_advect_q_and_solve_src, op_fvm, dt)

        else:
            split.strang(self, op_advect_q, op_solve_src, dt)

    af.eval(self.f)
    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return

def swss_step(self, dt):
    """
    Advances the system using a SWSS-split scheme. 
    This scheme is 2nd order accurate in time.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self.dt = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.fields_enabled == True
          ):
            split.swss(self, op_fvm, op_fields, dt)

        else:
            op_fvm(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.fields_enabled == True):
            
            def op_advect_q_and_solve_src(self, dt):
                
                return(split.swss(self, 
                                  op1 = op_advect_q,
                                  op2 = op_solve_src, 
                                  dt = dt
                                 )
                      )

            if(self.physical_system.params.solver_method_in_p == 'ASL'):
                split.swss(self, op_advect_q_and_solve_src, op_fields, dt)

            # For FVM in p-space:
            else: 
                split.swss(self, op_advect_q_and_solve_src, op_fvm, dt)

        else:
            split.swss(self, op_advect_q, op_solve_src, dt)

    af.eval(self.f)
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
    reference:<https://www.sciencedirect.com/science/article/pii/S089571771000436X>

    NOTE: This scheme is computationally expensive, and should only be used for testing/debugging

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self.dt = dt

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        if(    self.physical_system.params.solver_method_in_p == 'ASL'
           and self.physical_system.params.fields_enabled == True
          ):
            split.jia(self, op_fvm, op_fields, dt)

        else:
            op_fvm(self, dt)

    # Advective Semi-lagrangian method
    elif(self.physical_system.params.solver_method_in_q == 'ASL'):

        if(self.physical_system.params.fields_enabled == True):
            
            def op_advect_q_and_solve_src(self, dt):
                
                return(split.jia(self, 
                                 op1 = op_advect_q,
                                 op2 = op_solve_src, 
                                 dt = dt
                                )
                      )

            if(self.physical_system.params.solver_method_in_p == 'ASL'):
                split.jia(self, op_advect_q_and_solve_src, op_fields, dt)

            # For FVM in p-space:
            else: 
                split.jia(self, op_advect_q_and_solve_src, op_fvm, dt)

        else:
            split.jia(self, op_advect_q, op_solve_src, dt)

    af.eval(self.f)
    check_divergence(self)
    self.time_elapsed += dt 

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic

    return
