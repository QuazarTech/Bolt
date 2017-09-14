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


def strang_step(self, dt, performance_test_flag = False):
    """
    Advances the system using a strang-split 
    scheme. This scheme is 2nd order accurate in
    time.

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    if(performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_interp2            = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()

    # For hydrodynamic cases:
    if(self.physical_system.params.charge_electron == 0):
        # Advection in position space:
        f_interp_2d(self, 0.5 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Solving the source/sink terms:
        RK2_step(self, dt, False, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Advection in position space:
        f_interp_2d(self, 0.5 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

    else:
        # Advection in position space:
        f_interp_2d(self, 0.25 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Solving the source/sink terms:
        RK2_step(self, 0.5 * dt, False, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Advection in position space:
        f_interp_2d(self, 0.25 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Advection in velocity space:
        fields_step(self, dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Advection in position space:
        f_interp_2d(self, 0.25 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)
        
        # Solving the source/sink terms:
        RK2_step(self, 0.5 * dt, False, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)
        
        # Advection in position space:
        f_interp_2d(self, 0.25 * dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

    af.eval(self.f)

    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
    return


def lie_step(self, dt, performance_test = False):
    """
    Advances the system using a lie-split 
    scheme. This scheme is 1st order accurate in
    time.

    Parameters
    ----------

    dt : float
         Time-step size to evolve the system
    """
    if(performance_test_flag == True):
        
        if hasattr(self, 'time_ts'):
            pass
            
        else:
            self.time_ts                 = 0
            self.time_interp2            = 0
            self.time_fieldstep          = 0
            self.time_fieldsolver        = 0
            self.time_interp3            = 0
            self.time_sourcets           = 0
            self.time_communicate_f      = 0
            self.time_communicate_fields = 0

        tic = af.time()
    
    # For hydrodynamic cases:
    if(self.physical_system.params.charge_electron == 0):
        # Advection in position space:
        f_interp_2d(self, dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Solving the source/sink terms:
        RK2_step(self, dt, False, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

    else:
        # Advection in position space:
        f_interp_2d(self, dt, performance_test_flag)

        # Solving the source/sink terms:
        RK2_step(self, dt, False, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

        # Advection in velocity space:
        fields_step(self, dt, performance_test_flag)
        self._communicate_distribution_function(performance_test_flag)

    af.eval(self.f)
    
    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_ts += toc - tic
    
    return
