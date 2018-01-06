#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from .electrostatic import fft_poisson
from .fdtd_explicit import fdtd, fdtd_grid_to_ck_grid
from .. import interpolation_routines

def fields_step(self, dt):

    if(self.performance_test_flag == True):
        tic = af.time()
    
    if (self.physical_system.params.fields_solver == 'fft'):
        fft_poisson(self)
        self._communicate_fields()
        self._apply_bcs_fields()

    elif (self.physical_system.params.fields_solver == 'fdtd'):
        
        self.J1 =   af.sum(self.physical_system.params.charge) \
                  * self.compute_moments('mom_v1_bulk')  # (i + 1/2, j + 1/2)
        self.J2 =   af.sum(self.physical_system.params.charge) \
                  * self.compute_moments('mom_v2_bulk')  # (i + 1/2, j + 1/2)
        self.J3 =   af.sum(self.physical_system.params.charge) \
                  * self.compute_moments('mom_v3_bulk')  # (i + 1/2, j + 1/2)

        self.J1 = af.sum(self.J1, 1)
        self.J2 = af.sum(self.J2, 1)
        self.J3 = af.sum(self.J3, 1)

        # Obtaining the values for current density on the Yee-Grid:
        self.J1 = 0.5 * (self.J1 + af.shift(self.J1, 0, 0, 1))  # (i + 1/2, j)
        self.J2 = 0.5 * (self.J2 + af.shift(self.J2, 0, 1, 0))  # (i, j + 1/2)

        self.J3 = 0.25 * (  self.J3 + af.shift(self.J3, 0, 1, 0) +
                          + af.shift(self.J3, 0, 0, 1)
                          + af.shift(self.J3, 0, 1, 1)
                         )  # (i, j)

        # Here:
        # cell_centered_EM_fields[:3] is at n
        # cell_centered_EM_fields[3:] is at n+1/2
        # cell_centered_EM_fields_at_n_plus_half[3:] is at n-1/2

        self.cell_centered_EM_fields_at_n[:3] = self.cell_centered_EM_fields[:3]
        self.cell_centered_EM_fields_at_n[3:] = \
            0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[3:] 
                   + self.cell_centered_EM_fields[3:]
                  )

        self.cell_centered_EM_fields_at_n_plus_half[3:] = self.cell_centered_EM_fields[3:]

        fdtd(self, dt)
        fdtd_grid_to_ck_grid(self)

        # Here
        # cell_centered_EM_fields[:3] is at n+1
        # cell_centered_EM_fields[3:] is at n+3/2

        self.cell_centered_EM_fields_at_n_plus_half[:3] = \
            0.5 * (  self.cell_centered_EM_fields_at_n[:3] 
                   + self.cell_centered_EM_fields[:3]
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
