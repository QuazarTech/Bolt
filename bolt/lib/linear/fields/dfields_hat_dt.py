#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def dfields_hat_dt(fields_hat, self):
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

      fields_hat  : Fourier mode values for the fields at which the slope is computed
                    At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    df_dt : The time-derivative of f_hat
    """

    E1_hat = fields_hat[0]
    E2_hat = fields_hat[1]
    E3_hat = fields_hat[2]

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    # Equations Solved:
    # dE1/dt = + dB3/dq2 - J1
    # dE2/dt = - dB3/dq1 - J2
    # dE3/dt = dB2/dq1 - dB1/dq2 - J3

    dE1_hat_dt =  B3_hat * 1j * self.k_q2 - self.fields_solver.J1_hat
    dE2_hat_dt = -B3_hat * 1j * self.k_q1 - self.fields_solver.J2_hat
    dE3_hat_dt =  B2_hat * 1j * self.k_q1 - B1_hat * 1j * self.k_q2 - self.fields_solver.J3_hat

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
