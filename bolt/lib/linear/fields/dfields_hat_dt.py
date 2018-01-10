#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def dfields_hat_dt_multimode_evolution(fields_hat, self):

    E1_hat = fields_hat[0]
    E2_hat = fields_hat[1]
    E3_hat = fields_hat[2]

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    dE1_hat_dt =  B3_hat * 1j * self.k_q2 - self.fields_solver.J1_hat
    dE2_hat_dt = -B3_hat * 1j * self.k_q1 - self.fields_solver.J2_hat
    dE3_hat_dt =  B2_hat * 1j * self.k_q1 - B1_hat * 1j * self.k_q2 - self.fields_solver.J3_hat

    dB1_hat_dt = - E3_hat * 1j * self.k_q2
    dB2_hat_dt =   E3_hat * 1j * self.k_q1
    dB3_hat_dt =   E1_hat * 1j * self.k_q2 - E2_hat * 1j * self.k_q1

    dfields_hat_dt = af.join(0, 
                             af.join(0, dE1_hat_dt, dE2_hat_dt, dE3_hat_dt),
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
