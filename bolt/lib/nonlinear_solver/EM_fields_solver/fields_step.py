#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from .electrostatic import fft_poisson, compute_electrostatic_fields
from .fdtd_explicit import fdtd, fdtd_grid_to_ck_grid
from .. import interpolation_routines

def fields_step(self, dt):
    if(self.performance_test_flag == True):
        tic = af.time()
    
    if (self.physical_system.params.fields_solver == 'fft'):
        self._communicate_fields()
        fft_poisson(self)

    elif (self.physical_system.params.fields_solver == 'electrostatic'):
        self._communicate_fields()
        compute_electrostatic_fields(self)

    elif (self.physical_system.params.fields_solver == 'fdtd'):
        # Will returned a flattened array containing the values of
        # J1,2,3 in 2D space:
        self.J1 =   self.physical_system.params.charge_electron \
                  * self.compute_moments('mom_p1_bulk')  # (i + 1/2, j + 1/2)
        self.J2 =   self.physical_system.params.charge_electron \
                  * self.compute_moments('mom_p2_bulk')  # (i + 1/2, j + 1/2)
        self.J3 =   self.physical_system.params.charge_electron \
                  * self.compute_moments('mom_p3_bulk')  # (i + 1/2, j + 1/2)

        # Obtaining the values for current density on the Yee-Grid:
        self.J1 = 0.5 * (self.J1 + af.shift(self.J1, 0, 0, 1))  # (i + 1/2, j)
        self.J2 = 0.5 * (self.J2 + af.shift(self.J2, 0, 1, 0))  # (i, j + 1/2)

        self.J3 = 0.25 * (  self.J3 + af.shift(self.J3, 0, 1, 0) +
                          + af.shift(self.J3, 0, 0, 1)
                          + af.shift(self.J3, 0, 1, 1)
                         )  # (i, j)

        # Storing the values for the previous half-time step:
        # We do this since the B values on the CK grid are defined at
        # time t = n. While the B values on the FDTD grid are defined
        # at t = n + 1/2:
        self.B_fields_at_nth_timestep = self.cell_centered_EM_fields[3:]

        fdtd(self, dt)
        fdtd_grid_to_ck_grid(self)

        self.cell_centered_EM_fields[3:] = 0.5 * (  self.cell_centered_EM_fields[3:]
                                                  + self.B_fields_at_nth_timestep
                                                 )

    else:
        raise NotImplementedError('The method specified is \
                                   invalid/not-implemented'
                                 )

    interpolation_routines.f_interp_p_3d(self, dt)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldstep += toc - tic
    
    return
