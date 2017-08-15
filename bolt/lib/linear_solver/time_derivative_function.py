#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

from bolt.lib.linear_solver.EM_fields_solver \
    import compute_electrostatic_fields


@af.broadcast
def df_dt(self, f_hat):
    """
    Returns the value of the derivative of the mode perturbation of the
    distribution function with respect to time. This is used to evolve 
    the system with time.

    Input:
    ------

      f_hat :The vector for which the derivative is to be calculated 
             At t = 0 the initial state of the system is passed to this function.

    Output:
    -------
    df_dt : The time-derivatives of the distribution function f
    """
    self.f_hat = f_hat

    # If the system being solver for is an electrostatic case, the
    # Poisson equation can be solved for to obtain the mode perturbation
    # of the field quantities:
    if(self.physical_system.params.fields_solver == 'electrostatic' or
       self.physical_system.params.fields_solver == 'fft'
       ):
        compute_electrostatic_fields(self)

    # Scaling Appropriately:
    f = af.ifft2(0.5 * self.N_q2 * self.N_q1 * self.f_hat)

    C_f_hat = 2 * af.fft2(self._source_or_sink(f, self.q1_center, 
                                               self.q2_center, self.p1, 
                                               self.p2, self.p3,
                                               self.compute_moments,
                                               self.physical_system.params)) \
            / (self.N_q2 * self.N_q1)

    # Obtaining the advection terms in p-space:
    (A_p1,
     A_p2,
     A_p3) = self._A_p(self.q1_center, self.q2_center,
                       self.p1, self.p2, self.p3,
                       self.E1_hat, self.E2_hat, self.E3_hat,
                       self.B1_hat, self.B2_hat, self.B3_hat,
                       self.physical_system.params)

    fields_term = (A_p1 * self.dfdp1_background +
                   A_p2 * self.dfdp2_background +
                   A_p3 * self.dfdp3_background)

    df_hat_dt = -1j * (self.k_q1 * self._A_q1 +
                       self.k_q2 * self._A_q2) * self.f_hat

    # Addition of the field term only in the case where charge is non-zero:
    if(self.physical_system.params.charge_electron != 0):
        df_hat_dt -= fields_term

    # Similarly the collisional term is added only when tau != infinity:
    df_hat_dt += af.select(self.physical_system.params.tau(self.q1_center,
                                                           self.q2_center,
                                                           self.p1,
                                                           self.p2,
                                                           self.p3) != np.inf,
                           C_f_hat, 0)


    af.eval(df_hat_dt)

    del fields_term, A_p1, A_p2, A_p3, C_f_hat, f;af.device_gc()

    return(df_hat_dt)

@af.broadcast
def dY_dt(self, Y):
    """
    Returns the value of the derivative of the mode perturbation of the
    the field quantities with respect to time. This is used to evolve the
    system with time.

    Input:
    ------

      Y  : The array Y is the state of the system as given by the result of
           the last time-step's integration. Y is declared as a 3D array
           such that the elements of Y hold the following data:

           E1_hat = Y[:, :, 0]
           E2_hat = Y[:, :, 1]
           E3_hat = Y[:, :, 2]
           B1_hat = Y[:, :, 3]
           B2_hat = Y[:, :, 4]
           B3_hat = Y[:, :, 5]

           At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    dY_dt : The time-derivatives of all the quantities stored in Y
    """
    self.E1_hat = Y[:, :, 0]
    self.E2_hat = Y[:, :, 1]
    self.E3_hat = Y[:, :, 2]
    self.B1_hat = Y[:, :, 3]
    self.B2_hat = Y[:, :, 4]
    self.B3_hat = Y[:, :, 5]

    mom_bulk_p1 = self.compute_moments('mom_p1_bulk')
    mom_bulk_p2 = self.compute_moments('mom_p2_bulk')
    mom_bulk_p3 = self.compute_moments('mom_p3_bulk')

    # Scaling Appropriately:
    J1_hat = 2 * af.fft2(self.physical_system.params.charge_electron *
                         mom_bulk_p1) / (self.N_q1 * self.N_q2)
    J2_hat = 2 * af.fft2(self.physical_system.params.charge_electron *
                         mom_bulk_p2) / (self.N_q1 * self.N_q2)
    J3_hat = 2 * af.fft2(self.physical_system.params.charge_electron *
                         mom_bulk_p3) / (self.N_q1 * self.N_q2)

    # Solving the EM equations:

    # dE1/dt = + dB3/dq2
    # dE2/dt = - dB3/dq1
    # dE3/dt = dB2/dq1 - dB1/dq2

    dE1_hat_dt = (self.B3_hat * 1j * self.k_q2) - J1_hat
    dE2_hat_dt = (- self.B3_hat * 1j * self.k_q1) - J2_hat
    dE3_hat_dt = (self.B2_hat * 1j * self.k_q1 -
                  self.B1_hat * 1j * self.k_q2) - J3_hat

    # dB1/dt = - dE3/dq2
    # dB2/dt = + dE3/dq1
    # dB3/dt = - (dE2/dq1 - dE1/dq2)

    dB1_hat_dt = (- self.E3_hat * 1j * self.k_q2)
    dB2_hat_dt = (self.E3_hat * 1j * self.k_q1)
    dB3_hat_dt = (self.E1_hat * 1j * self.k_q2 - self.E1_hat * 1j * self.k_q1)

    # Declaration of the 4D array that is used to hold the derivative values:
    dY_dt = af.constant(0, self.p1.shape[0], self.p1.shape[1],
                        self.p1.shape[2], 7, dtype=af.Dtype.c64
                        )

    dY_dt[:, :, :, 0] = df_hat_dt

    dY_dt[:, :, :, 1] = dE1_hat_dt
    dY_dt[:, :, :, 2] = dE2_hat_dt
    dY_dt[:, :, :, 3] = dE3_hat_dt

    dY_dt[:, :, :, 4] = dB1_hat_dt
    dY_dt[:, :, :, 5] = dB2_hat_dt
    dY_dt[:, :, :, 6] = dB3_hat_dt

    return(dY_dt)
