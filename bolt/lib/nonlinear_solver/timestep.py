#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

# Importing functions used used for time-splitting and time-stepping:
import bolt.lib.nonlinear_solver.operatorsplitting_methods as split
import bolt.lib.timesteppers as timestepper

# Importing solver functions:
from bolt.lib.nonlinear_solver.FVM_solver.df_dt import df_dt
from bolt.lib.nonlinear_solver.interpolation_routines \
    import f_interp_2d, f_fft_interp_2d
from bolt.lib.nonlinear_solver.timestepper_source \
    import RK2_step, RK4_step
from bolt.lib.nonlinear_solver.EM_fields_solver.fields_step \
    import fields_step

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
    if(self.performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_interp2            = 0
            self.time_fvm_ts             = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_apply_bcs_f        = 0
            self.time_apply_bcs_fields   = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()

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

    # When solving using FVM in q-space:
    op_fvm_q     = timestepper.RK2(df_dt(df_dt, self.f))

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        split.strang(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(split.strang(self, op1 = op_advect_q,
                                op2 = op_solve_src, dt = dt
                               )
                  )
        split.strang(self, compound_op, op_fields, dt)

    self.time_elapsed += dt

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
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
    if(self.performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_interp2            = 0
            self.time_fvm_ts             = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_apply_bcs_f        = 0
            self.time_apply_bcs_fields   = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()

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
    if(self.performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_fvm_ts             = 0
            self.time_interp2            = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_apply_bcs_f        = 0
            self.time_apply_bcs_fields   = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()

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

    self.time_elapsed += dt

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
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
    if(self.performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_fvm_ts             = 0
            self.time_interp2            = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_apply_bcs_f        = 0
            self.time_apply_bcs_fields   = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()

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

    self.time_elapsed += dt

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
    return
