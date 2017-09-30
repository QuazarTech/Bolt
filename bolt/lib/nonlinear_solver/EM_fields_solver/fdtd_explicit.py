#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af


def fdtd(self, dt):
    # E's and B's are staggered in time such that
    # B's are defined at (n + 1/2), and E's are defined at n

    # Positions of grid point where field quantities are defined:
    # B1 --> (i, j + 1/2)
    # B2 --> (i + 1/2, j)
    # B3 --> (i + 1/2, j + 1/2)

    # E1 --> (i + 1/2, j)
    # E2 --> (i, j + 1/2)
    # E3 --> (i, j)

    # J1 --> (i + 1/2, j)
    # J2 --> (i, j + 1/2)
    # J3 --> (i, j)

    # The communicate function transfers the data from the local vectors
    # to the global vectors, in addition to dealing with the
    # boundary conditions:
    self._communicate_fields(True)
    
    if(self.performance_test_flag == True):
        tic = af.time()
    
    dq1 = self.dq1
    dq2 = self.dq2

    B1 = self.B1_fdtd
    B2 = self.B2_fdtd
    B3 = self.B3_fdtd

    # dE1/dt = + dB3/dq2
    # dE2/dt = - dB3/dq1
    # dE3/dt = dB2/dq1 - dB1/dq2

    B1_shifted_q2 = af.shift(B1, 0, 1)

    B2_shifted_q1 = af.shift(B2, 1, 0)

    B3_shifted_q1 = af.shift(B3, 1, 0)
    B3_shifted_q2 = af.shift(B3, 0, 1)

    self.E1_fdtd +=   (dt / dq2) * (B3 - B3_shifted_q2) - self.J1 * dt
    self.E2_fdtd +=  -(dt / dq1) * (B3 - B3_shifted_q1) - self.J2 * dt
    self.E3_fdtd +=   (dt / dq1) * (B2 - B2_shifted_q1) \
                    - (dt / dq2) * (B1 - B1_shifted_q2) \
                    - dt * self.J3
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    self._communicate_fields(True)

    if(self.performance_test_flag == True):
        tic = af.time()

    E1 = self.E1_fdtd
    E2 = self.E2_fdtd
    E3 = self.E3_fdtd

    # dB1/dt = - dE3/dq2
    # dB2/dt = + dE3/dq1
    # dB3/dt = - (dE2/dq1 - dE1/dq2)

    E1_shifted_q2 = af.shift(E1, 0, -1)

    E2_shifted_q1 = af.shift(E2, -1, 0)

    E3_shifted_q1 = af.shift(E3, -1, 0)
    E3_shifted_q2 = af.shift(E3, 0, -1)

    self.B1_fdtd += -(dt / dq2) * (E3_shifted_q2 - E3)
    self.B2_fdtd +=  (dt / dq1) * (E3_shifted_q1 - E3)
    self.B3_fdtd += - (dt / dq1) * (E2_shifted_q1 - E2) \
                    + (dt / dq2) * (E1_shifted_q2 - E1)

    af.eval(self.E1_fdtd, self.E2_fdtd, self.E3_fdtd,
            self.B1_fdtd, self.B2_fdtd, self.B3_fdtd
           )

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return


def fdtd_grid_to_ck_grid(self):

    # Interpolating at the (i + 1/2, j + 1/2) point of the grid to use for the
    # nonlinear solver:
    self.E1 = 0.5 * (self.E1_fdtd + af.shift(self.E1_fdtd,  0, -1))
    self.B1 = 0.5 * (self.B1_fdtd + af.shift(self.B1_fdtd, -1,  0))

    self.E2 = 0.5 * (self.E2_fdtd + af.shift(self.E2_fdtd, -1,  0))
    self.B2 = 0.5 * (self.B2_fdtd + af.shift(self.B2_fdtd,  0, -1))

    self.E3 = 0.25 * (  self.E3_fdtd + af.shift(self.E3_fdtd, 0, -1)
                      + af.shift(self.E3_fdtd, -1,  0)
                      + af.shift(self.E3_fdtd, -1, -1)
                     )

    self.B3 = self.B3_fdtd

    af.eval(self.E1, self.E2, self.E3, self.B1, self.B2, self.B3)
    return
