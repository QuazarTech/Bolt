#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
from numpy.fft import fftfreq

@af.broadcast
def multiply(a, b):
    return(a*b)

def fft_poisson(self, f=None):
    """
    Solves the Poisson Equation using the FFTs:
    Used as a backup solver in case of low resolution runs
    (ie. used on a single node) with periodic boundary
    conditions.
    """
    if(self.nls.performance_test_flag == True):
        tic = af.time()

    if (self.nls._comm.size != 1):
        raise Exception('FFT solver can only be used when run in serial')

    else:

        N_g_q = self.nls.N_ghost_q

        n = self.nls.compute_moments('density', f)[:, :, N_g_q:-N_g_q,
                                                   N_g_q:-N_g_q
                                                  ]
            
        # Reorder from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s) 
        rho = af.reorder(multiply(self.nls.physical_system.params.charge, (n - af.mean(n))),
                         2, 3, 0, 1
                        )

        # Summing for all the species:
        rho = af.sum(rho, 3)

        k_q1 = fftfreq(rho.shape[0], self.nls.dq1)
        k_q2 = fftfreq(rho.shape[1], self.nls.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q1 = af.to_array(k_q1)
        k_q2 = af.to_array(k_q2)

        rho_hat = af.fft2(rho)

        potential_hat       = rho_hat / (4 * np.pi**2 * (k_q1**2 + k_q2**2))
        potential_hat[0, 0] = 0

        E1_hat = -1j * 2 * np.pi * k_q1 * potential_hat
        E2_hat = -1j * 2 * np.pi * k_q2 * potential_hat

        # Non-inclusive of ghost-zones:
        E1_physical = af.reorder(af.real(af.ifft2(E1_hat)), 2, 3, 0, 1)
        E2_physical = af.reorder(af.real(af.ifft2(E2_hat)), 2, 3, 0, 1)

        self.cell_centered_EM_fields[0, 0, N_g_q:-N_g_q, N_g_q:-N_g_q] = E1_physical
        self.cell_centered_EM_fields[1, 0, N_g_q:-N_g_q, N_g_q:-N_g_q] = E2_physical

        af.eval(self.cell_centered_EM_fields)

    if(self.nls.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
