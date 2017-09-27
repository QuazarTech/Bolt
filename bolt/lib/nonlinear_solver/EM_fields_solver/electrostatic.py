#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import arrayfire as af
import numpy as np
from numpy.fft import fftfreq


class poisson_eqn(object):
    """
    This user class is an application context for the problem at hand;
    It contains some parametes and frames the matrix system depending on
    the system state. The Poisson2D object is used by the
    compute_electrostatic_fields function in computing the electrostatic fields
    using the PETSc's KSP solver methods
    """

    def __init__(self, nonlinear_solver_obj):
        self.da        = nonlinear_solver_obj._da_snes
        self.obj       = nonlinear_solver_obj
        self.local_phi = self.da.createLocalVec() # phi with ghost zones

    def compute_residual(self, snes, phi, residual):

        #self.da.globalToLocal(phi, self.local_phi)

        #phi_array      = self.local_phi.getArray(readonly=1)
        phi_array      = phi.getArray(readonly=1)
        residual_array = residual.getArray(readonly=0)
    
        residual_array[:] = phi_array**2. - 2.
        return

def compute_electrostatic_fields(self, performance_test_flag = False):
    
    if(performance_test_flag == True):
        tic = af.time()

    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_snes.getCorners()

    snes = PETSc.SNES().create()
    pde  = poisson_eqn(self)
    snes.setFunction(pde.compute_residual, self.glob_residual)
    
    snes.setDM(self._da_snes)
    snes.setFromOptions()
    snes.solve(None, self.glob_phi)
    
    phi_array = self.glob_phi.getArray()
    print("phi = ", phi_array)

    

#    pde = Poisson2D(self)
#    phi = self._da_ksp.createGlobalVec()
#    rho = self._da_ksp.createGlobalVec()
#
#    phi_local = self._da_ksp.createLocalVec()
#
#    A = PETSc.Mat().createPython([phi.getSizes(), rho.getSizes()],
#                                 comm=self._da_ksp.comm
#                                )
#    A.setPythonContext(pde)
#    A.setUp()
#
#    ksp = PETSc.KSP().create()
#
#    ksp.setOperators(A)
#    ksp.setType('cg')
#
#    pc = ksp.getPC()
#    pc.setType('none')
#
#    N_g = self.N_ghost
#    ksp.setTolerances(atol=1e-7)
#    pde.RHS(rho,
#              self.physical_system.params.charge_electron
#            * np.array(self.compute_moments('density')[N_g:-N_g,
#                                                       N_g:-N_g
#                                                      ]
#                       - 1
#                      )
#           )
#
#    ksp.setFromOptions()
#    ksp.solve(rho, phi)
#
#    num_tries = 0
#    while(ksp.converged is not True):
#        
#        ksp.setTolerances(atol = 10**(-6+num_tries), rtol = 10**(-6+num_tries))
#        ksp.solve(rho, phi)
#        num_tries += 1
#        
#        if(num_tries == 5):
#            raise Exception('KSP solver diverging!')
#
#    self._da_ksp.globalToLocal(phi, phi_local)

#    # Since rho was defined at (i + 0.5, j + 0.5)
#    # Electric Potential returned will also be at (i + 0.5, j + 0.5)
#    electric_potential = af.to_array(np.swapaxes(phi_local[:].
#                                                 reshape(  N_q2_local
#                                                         + 2 * self.N_ghost,
#                                                           N_q1_local +
#                                                         + 2 * self.N_ghost
#                                                        ),
#                                                 0, 1
#                                                )
#                                    )
#
#    # Obtaining the values at (i+0.5, j+0.5):
#    self.E1 = -(  af.shift(electric_potential, -1)
#                - af.shift(electric_potential,  1)
#               ) / (2 * self.dq1)
#
#    self.E2 = -(  af.shift(electric_potential, 0, -1)
#                - af.shift(electric_potential, 0,  1)
#               ) / (2 * self.dq2)
#
#    af.eval(self.E1, self.E2)

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

        E1_hat = -1j * 2 * np.pi * (k_q1) * potential_hat
        E2_hat = -1j * 2 * np.pi * (k_q2) * potential_hat

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
