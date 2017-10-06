#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test we check that the 2D Poisson solver
works as intended. For this purpose, we assign
a density distribution for which the analytical
solution for electrostatic fields may be computed.

This solution is then checked against the solution
given by the KSP solver
"""

import numpy as np
import arrayfire as af
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import pylab as pl


from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields

def compute_moments_sinusoidal(self, *args):
    rho = 1. + 0.1*af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2)

    rho_mean = af.mean(rho)
    print("rho_mean = ", rho_mean)

    return(rho - rho_mean)

def compute_moments_gaussian(self, *args):
    q2_minus = 0.25
    q2_plus  = 0.75

    regulator = 20  # larger value makes the transition sharper

    rho = 1 + 0.5 * (  af.tanh(( self.q2 - q2_minus)*regulator) 
                     - af.tanh(( self.q2 - q2_plus )*regulator)
                    )

#    rho =  af.exp(-0.*(self.q1 - 0.5)**2./0.01 - (self.q2 - 0.5)**2./0.01)
#    rho = (self.q2 - 0.5)**2. + (self.q1 - 0.5)**2.

#    rho = 1. + 0.1*af.sin(2*np.pi*self.q2)

#    sigma = 0.1
#    rho =  af.exp(-(self.q1)**2./(2.*sigma**2.) -(self.q2)**2./(2.*sigma**2.)) \
#    	  * 1./ sigma**2. / (2. * np.pi)

    N_g = self.N_ghost
    net_charge = af.sum(rho[N_g:-N_g, N_g:-N_g]) * self.dq1 * self.dq2

    total_volume =   (self.q1_end - self.q1_start) \
                   * (self.q2_end - self.q2_start)

    rho_zero_net_charge = rho - net_charge/total_volume

    print("Initial net charge = ", net_charge)

    return(rho_zero_net_charge)

class test(object):
    def __init__(self, N_q1, N_q2):
        # Creating an object with necessary attributes:
        self.physical_system = type('obj', (object, ),
                                    {'params': type('obj', (object, ),
                                        {'charge_electron': -1})
                                    }
                                   )

        self.q1_start = 0.
        self.q2_start = 0.

        self.q1_end = 1.
        self.q2_end = 1.

        self.N_q1 = N_q1
        self.N_q2 = N_q2

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        #self.N_ghost = np.random.randint(3, 5)
        self.N_ghost = 3

        self.q1 = self.q1_start \
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q1 + self.N_ghost
                                    )
                    ) * self.dq1
        
        self.q2 = self.q2_start \
                  + (0.5 + np.arange(-self.N_ghost,
                                     self.N_q2 + self.N_ghost
                                    )
                    ) * self.dq2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
        self.q2, self.q1 = af.to_array(self.q2), af.to_array(self.q1)

        self.E1 = af.constant(0, self.q1.shape[0], self.q1.shape[1],
                              dtype=af.Dtype.f64
                             )

        self.E2 = self.E1.copy()
        self.E3 = self.E1.copy()

        self.B1 = self.E1.copy()
        self.B2 = self.E1.copy()
        self.B3 = self.E1.copy()

        self._comm = PETSc.COMM_WORLD.tompi4py()

        self._da_snes = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                            stencil_width=self.N_ghost,
                                            boundary_type=('periodic',
                                                           'periodic'),
                                            stencil_type=1,
                                            dof = 1
                                           )
        self.glob_residual = self._da_snes.createGlobalVec()
        self.glob_phi  = self._da_snes.createGlobalVec()
        self.glob_phi.set(0.)

    compute_moments = compute_moments_gaussian

def test_compute_electrostatic_fields_1():
    obj = test(70, 70)
    compute_electrostatic_fields(obj)

#    N = 7*np.array([2, 4, 6, 8, 10, 12])
#    error_E1 = np.zeros(N.size)
#    error_E2 = np.zeros(N.size)
#
#    for i in range(N.size):
#        obj = test(N[i], N[i])
#        compute_electrostatic_fields(obj)
#
#        E1_expected =    (0.1 / np.pi) \
#                       * af.cos(  2 * np.pi * obj.q1
#                                + 4 * np.pi * obj.q2
#                               )
#
#        E2_expected =   (0.2 / np.pi) \
#                      * af.cos(  2 * np.pi * obj.q1
#                               + 4 * np.pi * obj.q2
#                              )
#
#        N_g = obj.N_ghost
#
#        error_E1[i] = af.sum(af.abs(  obj.E1[N_g:-N_g, N_g:-N_g]
#                                    - E1_expected[N_g:-N_g, N_g:-N_g]
#                                   )
#                            ) / (obj.E1[N_g:-N_g, N_g:-N_g].elements())
#
#        error_E2[i] = af.sum(af.abs(  obj.E2[N_g:-N_g, N_g:-N_g]
#                                    - E2_expected[N_g:-N_g, N_g:-N_g]
#                                   )
#                            ) / (obj.E2[N_g:-N_g, N_g:-N_g].elements())
#
#    print("Error E1 = ", error_E1)
#    print("Error E2 = ", error_E2)
#
#    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
#    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
#
#    assert (abs(poly_E1[0] + 2) < 0.2)
#    assert (abs(poly_E2[0] + 2) < 0.2)

def test_compute_electrostatic_fields_2():

    N = 7*np.array([2, 4, 6, 8, 10, 12])
    N = 7*np.array([12])
    error_E1 = np.zeros(N.size)
    error_E2 = np.zeros(N.size)

    for i in range(N.size):
        obj = test(N[i], N[i])
        compute_electrostatic_fields(obj)

        E1_expected = 0 * obj.q1
        
        q2_minus = 0.25
        q2_plus  = 0.75

        E2_expected = -0.5/20 * (  af.log(af.cosh(( obj.q2 - q2_minus)*20)) 
                                 - af.log(af.cosh(( obj.q2 - q2_plus )*20))
                                ) 

        N_g = obj.N_ghost

        error_E1[i] = af.sum(af.abs(  obj.E1[N_g:-N_g, N_g:-N_g]
                                    - E1_expected[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (obj.E1[N_g:-N_g, N_g:-N_g].elements())

        error_E2[i] = af.sum(af.abs(  obj.E2[N_g:-N_g, N_g:-N_g]
                                    - E2_expected[N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (obj.E2[N_g:-N_g, N_g:-N_g].elements())

    print("Error E1 = ", error_E1)
    print("Error E2 = ", error_E2)

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)

    assert (abs(poly_E1[0] + 2) < 0.2)
    assert (abs(poly_E2[0] + 2) < 0.2)


test_compute_electrostatic_fields_2()
