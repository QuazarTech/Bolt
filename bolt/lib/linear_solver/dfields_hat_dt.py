#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import arrayfire as af
import numpy as np

from .EM_fields_solver import compute_electrostatic_fields

def dfields_hat_dt_multimode_evolution(fields_hat, self):
    
    if(self.physical_system.params.fields_solver == 'fft'):
        self.J1_hat, self.J2_hat, self.J3_hat = 0, 0, 0

    dE1_hat_dt =  self.B3_hat * 1j * self.k_q2 - self.J1_hat
    dE2_hat_dt = -self.B3_hat * 1j * self.k_q1 - self.J2_hat
    dE3_hat_dt =  self.B2_hat * 1j * self.k_q1 \
                 -self.B1_hat * 1j * self.k_q2 - self.J3_hat

    dB1_hat_dt = -self.E3_hat * 1j * self.k_q2
    dB2_hat_dt =  self.E3_hat * 1j * self.k_q1
    dB3_hat_dt =   self.E1_hat * 1j * self.k_q2 \
                 - self.E2_hat * 1j * self.k_q1

    dfields_hat_dt = af.join(2, af.join(2, dE1_hat_dt, dE2_hat_dt, dE3_hat_dt),
                             dB1_hat_dt, dB2_hat_dt, dB3_hat_dt
                            )

    af.eval(dfields_hat_dt)
    return(dfields_hat_dt)

def dfields_hat_dt(fields_hat, self):

    return(dfields_hat_dt_multimode_evolution(fields_hat, self))

    # if(self.single_mode_evolution == True):
    #     return(dY_dt_singlemode_evolution(Y, self))

    # else:
    #     return(dY_dt_multimode_evolution(Y, self))
