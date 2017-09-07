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
    # For hydrodynamic cases:
    if(self.physical_system.params.charge_electron == 0):
        pass
        # Advection in position space:
        f_interp_2d(self, 0.5 * dt)
        self._communicate_distribution_function()

        # Solving the source/sink terms:
        RK2_step(self, dt)
        self._communicate_distribution_function()

        # Advection in position space:
        f_interp_2d(self, 0.5 * dt)
        self._communicate_distribution_function()

    else:
        # Advection in position space:
        f_interp_2d(self, 0.25 * dt)
        self._communicate_distribution_function()

        # Solving the source/sink terms:
        RK2_step(self, 0.5 * dt)
        self._communicate_distribution_function()

        # Advection in position space:
        f_interp_2d(self, 0.25 * dt)
        self._communicate_distribution_function()

        # Advection in velocity space:
        fields_step(self, dt)
        self._communicate_distribution_function()

        # Advection in position space:
        f_interp_2d(self, 0.25 * dt)
        self._communicate_distribution_function()
        
        # Solving the source/sink terms:
        RK2_step(self, 0.5 * dt)
        self._communicate_distribution_function()
        
        # Advection in position space:
        f_interp_2d(self, 0.25 * dt)
        self._communicate_distribution_function()

    af.eval(self.f)
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
    # For hydrodynamic cases:
    if(self.physical_system.params.charge_electron == 0):
        # Advection in position space:
        f_interp_2d(self, dt)
        self._communicate_distribution_function()

        # Solving the source/sink terms:
        RK2_step(self, dt)
        self._communicate_distribution_function()

    # Advection in position space:
    f_interp_2d(self, dt)

    # Solving the source/sink terms:
    RK2_step(self, dt)
    self._communicate_distribution_function()

    # Advection in velocity space:
    fields_step(self, dt)
    self._communicate_distribution_function()

    af.eval(self.f)
    return
