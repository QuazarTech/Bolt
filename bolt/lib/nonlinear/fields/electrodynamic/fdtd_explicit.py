#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from bolt.lib.nonlinear.communicate import communicate_fields
from ..boundaries import apply_bcs_fields

def fdtd_evolve_E(self, dt):
    """
    Evolves electric fields from E^n --> E^{n + 1}
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    
    if(self.performance_test_flag == True):
        tic = af.time()

    eps = self.params.eps
    mu  = self.params.mu

    dq1 = self.dq1
    dq2 = self.dq2

    B1 = self.yee_grid_EM_fields[3] # (i + 1/2, j)
    B2 = self.yee_grid_EM_fields[4] # (i, j + 1/2)
    B3 = self.yee_grid_EM_fields[5] # (i, j)

    B1_plus_q2 = af.shift(B1, 0, 0, 0, -1)

    B2_plus_q1 = af.shift(B2, 0, 0, -1, 0)

    B3_plus_q1 = af.shift(B3, 0, 0, -1, 0)
    B3_plus_q2 = af.shift(B3, 0, 0, 0, -1)

    # dE/dt = (∇ x B) / με - J / ε
    
    # dE1/dt = + (1 / με) * dB3/dq2 - J1 / ε
    # dE2/dt = - (1 / με) * dB3/dq1 - J2 / ε
    # dE3/dt = (1 / με) * (dB2/dq1 - dB1/dq2) - J3 / ε

    # curlB_x =  dB3/dq2
    curlB_1 =  (B3_plus_q2 - B3) / dq2 # (i, j + 1/2)
    # curlB_y = -dB3/dq1
    curlB_2 = -(B3_plus_q1 - B3) / dq1 # (i + 1/2, j)
    # curlB_z = (dB2/dq1 - dB1/dq2)
    curlB_3 =  (B2_plus_q1 - B2) / dq1 - (B1_plus_q2 - B1) / dq2 # (i + 1/2, j + 1/2)

    if(self.params.hybrid_model_enabled == True):
        # This is already assigned under df_dt_fvm
        # Here we are just checking that J = (∇ x B) / μ
        assert(af.mean(af.abs(self.J1 - curlB_1 / mu)) < 1e-14)
        assert(af.mean(af.abs(self.J2 - curlB_2 / mu)) < 1e-14)
        assert(af.mean(af.abs(self.J3 - curlB_3 / mu)) < 1e-14)

    else:
        # E1 --> (i, j + 1/2)
        self.yee_grid_EM_fields[0] += (dt / (mu * eps)) * curlB_1 - self.J1 * dt / eps
        # E2 --> (i + 1/2, j)
        self.yee_grid_EM_fields[1] += (dt / (mu * eps)) * curlB_2 - self.J2 * dt / eps
        # E3 --> (i + 1/2, j + 1/2)
        self.yee_grid_EM_fields[2] += (dt / (mu * eps)) * curlB_3 - self.J3 * dt / eps

        # USED TO CHECK:
        # curlB1_plus_q1 = af.shift(curlB_1, 0, 0, -1)
        # curlB2_plus_q2 = af.shift(curlB_2, 0, 0, 0, -1)

        # divcurlB = (curlB1_plus_q1 - curlB_1) / self.dq1 + (curlB2_plus_q2 - curlB_2) / self.dq2
        # print('Divergence of curlB:', af.mean(af.abs(divcurlB)))

    af.eval(self.yee_grid_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic

    return

def fdtd_evolve_B(self, dt):
    """
    Evolves magnetic fields from B^{n + 1/2} --> B^{n + 3/2}
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    dq1 = self.dq1
    dq2 = self.dq2

    E1 = self.yee_grid_EM_fields[0] # (i, j + 1/2)
    E2 = self.yee_grid_EM_fields[1] # (i + 1/2, j)
    E3 = self.yee_grid_EM_fields[2] # (i + 1/2, j + 1/2)

    E1_minus_q2 = af.shift(E1, 0, 0, 0, 1)

    E2_minus_q1 = af.shift(E2, 0, 0, 1, 0)

    E3_minus_q1 = af.shift(E3, 0, 0, 1, 0)
    E3_minus_q2 = af.shift(E3, 0, 0, 0, 1)

    # dB/dt = -(∇ x E)

    # dB1/dt = - dE3/dq2
    # dB2/dt = + dE3/dq1
    # dB3/dt = - (dE2/dq1 - dE1/dq2)

    # curlE_x =  dE3/dq2
    curlE_1 =  (E3 - E3_minus_q2) / dq2 # (i + 1/2, j)
    # curlE_y = -dE3/dq1
    curlE_2 = -(E3 - E3_minus_q1) / dq1 # (i, j + 1/2)
    # curlE_z = (dE2/dq1 - dE1/dq2)
    curlE_3 =  (E2 - E2_minus_q1) / dq1 - (E1 - E1_minus_q2) / dq2 # (i, j)

    # B1 --> (i + 1/2, j)
    self.yee_grid_EM_fields[3] += -dt * curlE_1
    # B2 --> (i, j + 1/2)
    self.yee_grid_EM_fields[4] += -dt * curlE_2
    # B3 --> (i, j)
    self.yee_grid_EM_fields[5] += -dt * curlE_3

    af.eval(self.yee_grid_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic

    return


def fdtd(self, dt):
    """
    Evolves the EM fields variables on a Yee-Grid using FDTD:
    E's and B's are staggered in time such that
    B's are defined at (n + 1/2), and E's are defined at n

    Positions of grid point where field quantities are defined:
    B1 --> (i + 1/2, j)
    B2 --> (i, j + 1/2)
    B3 --> (i, j)

    E1 --> (i, j + 1/2)
    E2 --> (i + 1/2, j)
    E3 --> (i + 1/2, j + 1/2)

    J1 --> (i, j + 1/2)
    J2 --> (i + 1/2, j)
    J3 --> (i + 1/2, j + 1/2)
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    # The communicate function transfers the data from the 
    # local vectors to the global vectors, in addition to  
    # dealing with periodic boundary conditions:
    communicate_fields(self, True)
    apply_bcs_fields(self, True)
    fdtd_evolve_E(self, dt)
    
    communicate_fields(self, True)
    apply_bcs_fields(self, True)
    fdtd_evolve_B(self, dt)
    
    return
