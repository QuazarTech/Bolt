#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import arrayfire as af

from . import integrators
from .df_hat_dt import df_hat_dt
from .fields.dfields_hat_dt import dfields_hat_dt

from bolt.lib.utils.fft_funcs import fft2, ifft2

def RK5_step(self, dt):
    """
    Evolves the physical system defined using an RK5
    integrator. This method is 5th order accurate.

    Parameters
    ----------

    dt: double
        The timestep size.

    """
    # For purely collisional cases:
    if(self.physical_system.params.instantaneous_collisions == True):

        f0 = self._source(0.5 * self.N_q1 * self.N_q2 * af.real(ifft2(self.f_hat)),
                          self.time_elapsed, self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.compute_moments, 
                          self.physical_system.params, 
                          True
                         )

        self.f_hat = 2 * fft2(f0) / (self.N_q1 * self.N_q2)

    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        # Since the fields and the distribution function are coupled, 
        # we evolve the system by making use of a coupled integrator 
        # which ensures that throughout the timestepping they are 
        # evaluated at the same temporal locations.
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK5_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )
        af.eval(self.f_hat)
        af.eval(self.fields_solver.fields_hat)

    else:
        self.f_hat = integrators.RK5(df_hat_dt, self.f_hat,
                                     dt, self.fields_solver.fields_hat, self
                                    )
        af.eval(self.f_hat)
    
    return

def RK4_step(self, dt):
    """
    Evolves the physical system defined using an RK4
    integrator. This method is 4th order accurate.

    Parameters
    ----------
    dt: double
        The timestep size.
    """
    # For purely collisional cases:
    if(self.physical_system.params.instantaneous_collisions == True):

        f0 = self._source(0.5 * self.N_q1 * self.N_q2 * af.real(ifft2(self.f_hat)),
                          self.time_elapsed, self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.compute_moments, 
                          self.physical_system.params, 
                          True
                         )

        self.f_hat = 2 * fft2(f0) / (self.N_q1 * self.N_q2)

    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        # Since the fields and the distribution function are coupled, 
        # we evolve the system by making use of a coupled integrator 
        # which ensures that throughout the timestepping they are 
        # evaluated at the same temporal locations.
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK4_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )
        af.eval(self.f_hat)
        af.eval(self.fields_solver.fields_hat)

    else:
        self.f_hat = integrators.RK4(df_hat_dt, self.f_hat,
                                     dt, self.fields_solver.fields_hat, self
                                    )
        af.eval(self.f_hat)

    return

def RK2_step(self, dt):
    """
    Evolves the physical system defined using an RK2
    integrator. This method is 2nd order accurate.

    Parameters
    ----------
    dt: double
        The timestep size.
    """
    # For purely collisional cases:
    if(self.physical_system.params.instantaneous_collisions == True):

        f0 = self._source(0.5 * self.N_q1 * self.N_q2 * af.real(ifft2(self.f_hat)),
                          self.time_elapsed, self.q1_center, self.q2_center,
                          self.p1, self.p2, self.p3, 
                          self.compute_moments, 
                          self.physical_system.params, 
                          True
                         )

        self.f_hat = 2 * fft2(f0) / (self.N_q1 * self.N_q2)

    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        # Since the fields and the distribution function are coupled, 
        # we evolve the system by making use of a coupled integrator 
        # which ensures that throughout the timestepping they are 
        # evaluated at the same temporal locations.
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK2_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )
        af.eval(self.f_hat)
        af.eval(self.fields_solver.fields_hat)

    else:
        self.f_hat = integrators.RK2(df_hat_dt, self.f_hat,
                                     dt, self.fields_solver.fields_hat, self
                                    )
        af.eval(self.f_hat)

    return
