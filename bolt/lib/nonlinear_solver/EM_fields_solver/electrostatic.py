#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import petsc4py, sys 
petsc4py.init(sys.argv)
from petsc4py import PETSc
import arrayfire as af
import numpy as np
from numpy.fft import fftfreq


class Poisson2D(object):
    """
    This user class is an application context for the problem at hand;
    It contains some parametes and frames the matrix system depending on
    the system state. The Poisson2D object is used by the
    compute_electrostatic_fields function in computing the electrostatic fields
    using the PETSc's KSP solver methods
    """

    def __init__(self, obj):
        self.da     = obj._da_ksp
        self.obj    = obj
        self.localX = self.da.createLocalVec()

    def RHS(self, rho, rho_array):
        rho_val    = self.da.getVecArray(rho)
        rho_val[:] = rho_array

    def mult(self, mat, X, Y):

        self.da.globalToLocal(X, self.localX)

        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)

        (q1_start, q1_end), (q2_start, q2_end) = self.da.getRanges()

        for j in range(q1_start, q1_end):
            for i in range(q2_start, q2_end):
                u = x[j, i]  # center

                u_w = x[j, i - 1]  # west
                u_e = x[j, i + 1]  # east
                u_s = x[j - 1, i]  # south
                u_n = x[j + 1, i]  # north

                u_xx = (-u_e + 2 * u - u_w) / self.obj.dq2**2
                u_yy = (-u_n + 2 * u - u_s) / self.obj.dq1**2

                y[j, i] = u_xx + u_yy


def compute_electrostatic_fields(self, performance_test_flag = False):
    
    if(performance_test_flag == True):
        tic = af.time()

    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local,N_q2_local)) = self._da_ksp.getCorners()

    pde = Poisson2D(self)
    phi = self._da_ksp.createGlobalVec()
    rho = self._da_ksp.createGlobalVec()

    phi_local = self._da_ksp.createLocalVec()

    A = PETSc.Mat().createPython([phi.getSizes(), rho.getSizes()],
                                 comm=self._da_ksp.comm
                                )
    A.setPythonContext(pde)
    A.setUp()

    ksp = PETSc.KSP().create()

    ksp.setOperators(A)
    ksp.setFromOptions()
    # ksp.setType('cg')

    pc = ksp.getPC()
    pc.setType('none')

    N_g = self.N_ghost
    ksp.setTolerances(atol=1e-7)
    pde.RHS(rho,
              self.physical_system.params.charge_electron
            * np.array(self.compute_moments('density')[N_g:-N_g,
                                                       N_g:-N_g
                                                      ]
                       - 1
                      )
           )

    ksp.solve(rho, phi)

    num_tries = 0
    while(ksp.converged is not True):
        
        ksp.setTolerances(atol = 10**(-6+num_tries), rtol = 10**(-6+num_tries))
        ksp.solve(rho, phi)
        num_tries += 1
        
        if(num_tries == 5):
            raise Exception('KSP solver diverging!')

    self._da_ksp.globalToLocal(phi, phi_local)

    # Since rho was defined at (i + 0.5, j + 0.5)
    # Electric Potential returned will also be at (i + 0.5, j + 0.5)
    electric_potential = af.to_array(np.swapaxes(phi_local[:].
                                                 reshape(  N_q2_local
                                                         + 2 * self.N_ghost,
                                                           N_q1_local +
                                                         + 2 * self.N_ghost
                                                        ),
                                                 0, 1
                                                )
                                    )

    # Obtaining the values at (i+0.5, j+0.5):
    self.E1 = -(  af.shift(electric_potential, -1)
                - af.shift(electric_potential,  1)
               ) / (2 * self.dq1)

    self.E2 = -(  af.shift(electric_potential, 0, -1)
                - af.shift(electric_potential, 0,  1)
               ) / (2 * self.dq2)

    af.eval(self.E1, self.E2)

    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return


def fft_poisson(self, performance_test_flag = False):
    """
    Solves the Poisson Equation using the FFTs:

    Used as a backup solver in case of low resolution runs
    (ie. used on a single node) with periodic boundary
    conditions.
    """
    if(performance_test_flag == True):
        tic = af.time()

    if (self._comm.size != 1):
        raise Exception('FFT solver can only be used when run in serial')

    else:
        N_g = self.N_ghost
        rho =   self.physical_system.params.charge_electron \
              * (self.compute_moments('density')[N_g:-N_g, N_g:-N_g])

        k_q1 = fftfreq(rho.shape[0], self.dq1)
        k_q2 = fftfreq(rho.shape[1], self.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q1 = af.to_array(k_q1)
        k_q2 = af.to_array(k_q2)

        rho_hat = af.fft2(rho)

        potential_hat       = rho_hat / (4 * np.pi**2 * (k_q1**2 + k_q2**2))
        potential_hat[0, 0] = 0

        E1_hat = -1j * 2 * np.pi * k_q1 * potential_hat
        E2_hat = -1j * 2 * np.pi * k_q2 * potential_hat

        self.E1[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E1_hat))
        self.E2[N_g:-N_g, N_g:-N_g] = af.real(af.ifft2(E2_hat))

        # Applying Periodic B.C's:
        self._communicate_fields(performance_test_flag)
        af.eval(self.E1, self.E2)

    if(performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldsolver += toc - tic
    
    return
