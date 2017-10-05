#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

# Importing solver functions:
from bolt.lib.nonlinear_solver.interpolation_routines \
    import f_interp_2d
from bolt.lib.nonlinear_solver.timestepper_source \
    import RK2_step
from bolt.lib.nonlinear_solver.EM_fields_solver.fields_step \
    import fields_step

def _strang_split_operations(self, op1, op2, dt):
    """
    Performs strang splitting for any 2 operators.

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    """
    op1(self, 0.5 * dt)
    op2(self, dt)
    op1(self, 0.5 * dt)

    return    

def _lie_split_operations(self, op1, op2, dt):
    """
    Performs lie splitting for any 2 operators.

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    """
    op1(self, dt)
    op2(self, dt)

    return  

def _swss_split_operations(self, op1, op2, dt):
    """
    Performs SWSS splitting for any 2 operators.

    Parameters
    ----------
    self: object
          Nonlinear solver object which describes the system
          being evolved

    op1 : function
          Function which solves the 1st part of the split
          equation. Should only take solver object and dt 
          as arguments.
    
    op2 : function
          Function which solves the 2nd part of the split
          equation. Should only take the solver object and
          dt as arguments

    dt : float
         Time-step size to evolve the system
    
    """
    # Storing start values:
    f_start = self.f

    E1_start = self.E1
    E2_start = self.E2
    E3_start = self.E3
    
    B1_start = self.B1
    B2_start = self.B2
    B3_start = self.B3

    # Performing e^At e^Bt
    op1(self, dt)
    op2(self, dt)

    # Storing values obtained in this order:
    f_intermediate = self.f

    E1_intermediate = self.E1
    E2_intermediate = self.E2
    E3_intermediate = self.E3
    
    B1_intermediate = self.B1
    B2_intermediate = self.B2
    B3_intermediate = self.B3

    # Reassiging starting values:
    self.f = f_start    

    self.E1 = E1_start
    self.E2 = E2_start
    self.E3 = E3_start
    
    self.B1 = B1_start
    self.B2 = B2_start
    self.B3 = B3_start

    # Performing e^Bt e^At:
    op2(self, dt)
    op1(self, dt)

    # Averaging solution:
    self.f = 0.5 * (self.f + f_intermediate)
    
    self.E1 = 0.5 * (self.E1 + E1_intermediate)
    self.E2 = 0.5 * (self.E2 + E2_intermediate)
    self.E3 = 0.5 * (self.E3 + E3_intermediate)
    
    self.B1 = 0.5 * (self.B1 + B1_intermediate)
    self.B2 = 0.5 * (self.B2 + B2_intermediate)
    self.B3 = 0.5 * (self.B3 + B3_intermediate)

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
    if(self.performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
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
        f_interp_2d(self, dt)
        self._communicate_f()
        self._apply_bcs_f()

        return

    # Solving the source/sink terms:
    op_solve_src = RK2_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        _strang_split_operations(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(_strang_split_operations(self, op1 = op_advect_q,
                                            op2 = op_solve_src, dt = dt
                                           )
                  )
        _strang_split_operations(self, compound_op, op_fields, dt)

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
        f_interp_2d(self, dt)
        self._communicate_f()
        self._apply_bcs_f()

        return

    # Solving the source/sink terms:
    op_solve_src = RK2_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        _lie_split_operations(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(_lie_split_operations(self, op1 = op_advect_q,
                                         op2 = op_solve_src, dt = dt
                                        )
                  )
        _lie_split_operations(self, compound_op, op_fields, dt)

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
        f_interp_2d(self, dt)
        self._communicate_f()
        self._apply_bcs_f()

        return

    # Solving the source/sink terms:
    op_solve_src = RK2_step 
    # Solving for fields/advection in velocity space:
    op_fields    = fields_step

    # Cases which lack fields:
    if(self.physical_system.params.charge_electron == 0):
        _swss_split_operations(self, op_advect_q, op_solve_src, dt)
    
    
    else:
        def compound_op(self, dt):
            return(_swss_split_operations(self, op1 = op_advect_q,
                                          op2 = op_solve_src, dt = dt
                                         )
                  )
        _swss_split_operations(self, compound_op, op_fields, dt)

    self.time_elapsed += dt

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
    return
