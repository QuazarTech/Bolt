#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from bolt.lib.utils.fft_funcs import fft2, ifft2
from bolt.lib.utils.broadcasted_primitive_operations import multiply

def dfields_hat_dt(f_hat, fields_hat, self):
    """
    Returns the value of the derivative of the fields_hat with respect to time 
    respect to time. This is used to evolve the fields with time. 
    
    NOTE:All the fields quantities are included in fields_hat as follows:

    E1_hat = fields_hat[0]
    E2_hat = fields_hat[1]
    E3_hat = fields_hat[2]

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    Input:
    ------

      f_hat  : Fourier mode values for the distribution function at which the slope is computed
               At t = 0 the initial state of the system is passed to this function:

      fields_hat  : Fourier mode values for the fields at which the slope is computed
                    At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    df_dt : The time-derivative of f_hat
    """
    J1_hat = multiply(self.physical_system.params.charge,
                      self.compute_moments('mom_v1_bulk', f_hat=f_hat)
                     ) 
    J2_hat = multiply(self.physical_system.params.charge,
                      self.compute_moments('mom_v2_bulk', f_hat=f_hat)
                     ) 
    J3_hat = multiply(self.physical_system.params.charge,
                      self.compute_moments('mom_v3_bulk', f_hat=f_hat)
                     ) 

    eps = self.physical_system.params.eps
    mu  = self.physical_system.params.mu

    # Summing along all species:
    J1_hat = af.sum(J1_hat, 1)
    J2_hat = af.sum(J2_hat, 1)
    J3_hat = af.sum(J3_hat, 1)

    E1_hat = fields_hat[0]
    E2_hat = fields_hat[1]
    E3_hat = fields_hat[2]

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    # Checking that there is no mean field component:
    # try:
    #     assert(af.mean(af.abs(B1_hat[:, 0, 0])) < 1e-12)
    #     assert(af.mean(af.abs(B2_hat[:, 0, 0])) < 1e-12)
    #     assert(af.mean(af.abs(B3_hat[:, 0, 0])) < 1e-12)
    # except:
    #     raise SystemExit('Linear Solver cannot solve for non-zero mean magnetic fields')

    # Equations Solved:
    # dE1/dt = + dB3/dq2 - J1
    # dE2/dt = - dB3/dq1 - J2
    # dE3/dt = dB2/dq1 - dB1/dq2 - J3

    dE1_hat_dt =  B3_hat * 1j * self.k_q2 / (mu * eps) - J1_hat / eps
    dE2_hat_dt = -B3_hat * 1j * self.k_q1 / (mu * eps) - J2_hat / eps
    dE3_hat_dt =  (B2_hat * 1j * self.k_q1 - B1_hat * 1j * self.k_q2) / (mu * eps) \
                 -J3_hat / eps

    # dB1/dt = - dE3/dq2
    # dB2/dt = + dE3/dq1
    # dB3/dt = - (dE2/dq1 - dE1/dq2)

    dB1_hat_dt = -E3_hat * 1j * self.k_q2
    dB2_hat_dt =  E3_hat * 1j * self.k_q1
    dB3_hat_dt =  E1_hat * 1j * self.k_q2 - E2_hat * 1j * self.k_q1

    dfields_hat_dt = af.join(0, 
                             af.join(0, dE1_hat_dt, dE2_hat_dt, dE3_hat_dt),
                             dB1_hat_dt, dB2_hat_dt, dB3_hat_dt
                            )

    af.eval(dfields_hat_dt)
    return(dfields_hat_dt)
