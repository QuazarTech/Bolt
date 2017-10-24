#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

# Importing functions used used for time-splitting and time-stepping:
from .temporal_evolution import operator_splitting_methods as split
from .temporal_evolution import integrators

# Importing solver functions:
from .FVM_solver.df_dt import df_dt
from .interpolation_routines import f_interp_2d
from .EM_fields_solver.fields_step import fields_step

def op_fvm_q(self, source, dt):
    self._communicate_f()
    self._apply_bcs_f()
    self.f = integrators.RK2(df_dt, self.f, dt,  
                             self._A_q1, self._A_q2,
                             self.dq1, self.dq2, source
                            )

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

    # Making the source term only take the value of f as an argument
    # This is done to comply with the format that the integrators accept:
    def source(f):
        return(self._source(f, self.q1_center, self.q2_center,
                            self.p1, self.p2, self.p3, 
                            self.compute_moments, 
                            self.physical_system.params
                           )
              )

    op_fvm_q(self, source, dt)

    # if(self.physical_system.params.solver_method_in_q == 'FVM'):
 
    

    # Advective Semi-lagrangian method
    # elif(self.physical_system.params.solver_method_in_q == 'ASL'):
        
    #     def op_advect_q(self, dt):
    
    #         self._communicate_f()
    #         self._apply_bcs_f()
    #         f_interp_2d(self, dt)

    #     op_solve_src = integrators.RK2(source, self.f, dt)

    # if(self.physical_system.params.charge_electron == 0):
    # Advection in position space:
    # split.strang(self, op_advect_q, op_solve_src, dt)

    # Solving the source/sink terms:
    # Solving for fields/advection in velocity space:
    # op_fields    = fields_step
    # When solving using FVM in q-space:

    # Cases which lack fields:
    # else:
    #     def compound_op(self, dt):
    #         return(split.strang(self, op1 = op_advect_q,
    #                             op2 = op_solve_src, dt = dt
    #                            )
    #               )
    #     split.strang(self, compound_op, op_fields, dt)

    return


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

    # Advection in position space:

    def op_advect_q(self, dt):
        self._communicate_f()
        self._apply_bcs_f()
        f_interp_2d(self, dt)

        return

    # Solving the source/sink terms:
    op_solve_src = RK2_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        split.lie(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(split.lie(self, op1 = op_advect_q,
                                 op2 = op_solve_src, dt = dt
                                )
                  )
        split.lie(self, compound_op, op_fields, dt)

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

    # Advection in position space:
    def op_advect_q(self, dt):
        self._communicate_f()
        self._apply_bcs_f()
        f_interp_2d(self, dt)

        return

    # Solving the source/sink terms:
    op_solve_src = RK2_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        split.swss(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(split.swss(self, op1 = op_advect_q,
                                  op2 = op_solve_src, dt = dt
                                 )
                  )
        split.swss(self, compound_op, op_fields, dt)

    return

def jia_step(self, dt):
    """
    Advances the system using the Jia split scheme.

    NOTE: This scheme is computationally expensive, and is 
          primarily used for testing

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """

    # Advection in position space:
    def op_advect_q(self, dt):
        self._communicate_f()
        self._apply_bcs_f()
        f_interp_2d(self, dt)

        return

    # Solving the source/sink terms:
    op_solve_src = RK4_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        split.jia(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(split.jia(self, op1 = op_advect_q,
                                 op2 = op_solve_src, dt = dt
                                )
                  )
        split.jia(self, compound_op, op_fields, dt)

    return
