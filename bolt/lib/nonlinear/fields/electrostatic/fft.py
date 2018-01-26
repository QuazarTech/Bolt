#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

from bolt.lib.nonlinear.utils.broadcasted_primitive_operations import multiply

def fft_poisson(self, rho):
    """
    Solves the Poisson Equation using FFTs:
    Used as a backup solver for low resolution runs
    (ie. used on a single node) with periodic boundary
    conditions.

    Parameters
    ----------

    rho : af.Array
          Array that holds the charge density for each species

    """
    if(self.performance_test_flag == True):
        tic = af.time()

    if (self._comm.size != 1):
        raise Exception('FFT solver can only be used when run in serial')

    else:

        N_g = self.N_g
            
        # Reorder from (1, N_s, N_q1, N_q2) --> (N_q1, N_q2, 1, N_s) 
        rho = af.reorder(rho[:, :, N_g:-N_g, N_g:-N_g] , 2, 3, 0, 1)
        # Summing for all the species:
        rho = af.sum(rho, 3)

        k_q1 = np.fft.fftfreq(rho.shape[0], self.dq1)
        k_q2 = np.fft.fftfreq(rho.shape[1], self.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q1 = af.to_array(k_q1)
        k_q2 = af.to_array(k_q2)

        rho_hat = af.fft2(rho)

        potential_hat       = rho_hat / (4 * np.pi**2 * (k_q1**2 + k_q2**2))
        potential_hat[0, 0] = 0

        E1_hat = -1j * 2 * np.pi * k_q1 * potential_hat
        E2_hat = -1j * 2 * np.pi * k_q2 * potential_hat

        # Non-inclusive of ghost-zones:
        E1_ifft = af.ifft2(E1_hat, scale=1)/(E1_hat.shape[0] * E1_hat.shape[1])
        E2_ifft = af.ifft2(E2_hat, scale=1)/(E2_hat.shape[0] * E2_hat.shape[1])
        
        E1_physical = af.reorder(af.real(E1_ifft), 2, 3, 0, 1)
        E2_physical = af.reorder(af.real(E2_ifft), 2, 3, 0, 1)

        self.cell_centered_EM_fields[0, 0, N_g:-N_g, N_g:-N_g] = E1_physical
        self.cell_centered_EM_fields[1, 0, N_g:-N_g, N_g:-N_g] = E2_physical

        af.eval(self.cell_centered_EM_fields)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
