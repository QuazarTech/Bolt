#!/usr/bin/env python 
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import arrayfire as af
import numpy as np
from scipy.fftpack import fftfreq

# This user class is an application context for the problem at hand; 
# It contains some parametes and frames the matrix system depending on the system state
class Poisson2D(object):

  def __init__(self, obj):
    assert obj._da.getDim() == 2
    self.da     = obj._da_fields
    self.obj    = obj
    self.localX = self.da.createLocalVec()

  def formRHS(self, rho, rho_array):
    rho_val    = self.da.getVecArray(rho)
    rho_val[:] = rho_array * self.obj.dq1 * self.obj.dq2

  def mult(self, mat, X, Y):
        
    self.da.globalToLocal(X, self.localX)
    
    x = self.da.getVecArray(self.localX)
    y = self.da.getVecArray(Y)
    
    (i_q1_start, i_q1_end), (i_q2_start, i_q2_end) = self.da.getRanges()
    
    for j in range(i_q1_start, i_q1_end):
      for i in range(i_q2_start, i_q2_end):
        u   = x[j, i]   # center
        u_w = x[j, i-1] # west
        u_e = x[j, i+1] # east
        u_s = x[j-1, i] # south
        u_n = x[j+1, i] # north
        
        u_q1q1 = (-u_e + 2*u - u_w)*self.dq1/self.dq2
        u_q2q2 = (-u_n + 2*u - u_s)*self.dq1/self.dq2
 
        y[j, i] = u_q1q1 + u_q2q2

def solve_electrostatic_fields(self):
  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()

  pde = Poisson2D(self)
  phi = self._da_fields.createGlobalVec()
  rho = self._da_fields.createGlobalVec()

  phi_local = self._da.createLocalVec()

  A = PETSc.Mat().createPython([phi.getSizes(), rho.getSizes()], comm = self._da_fields.comm)
  A.setPythonContext(pde)
  A.setUp()

  ksp = PETSc.KSP().create()

  ksp.setOperators(A)
  ksp.setType('cg')

  pc = ksp.getPC()
  pc.setType('none')

  pde.formRHS(rho, np.array(self.compute_moments('density')))
  ksp.solve(rho, phi)

  self._da_fields.globalToLocal(phi, phi_local)

  # Since rho was defined at (i + 0.5, j + 0.5) 
  # Electric Potential returned will also be at (i + 0.5, j + 0.5)
  electric_potential = af.to_array(np.swapaxes(phi_local[:].reshape(N_q2_local + 2*self.N_ghost,\
                                                                    N_q1_local + 2*self.N_ghost
                                                                   ), 0, 1
                                              )
                                  )

  self.E1 = -(af.shift(electric_potential, -1)    - electric_potential)/self.dq1 #(i+1/2, j)
  self.E2 = -(af.shift(electric_potential, 0, -1) - electric_potential)/self.dq2 #(i, j+1/2)

  # Obtaining the values at (i+0.5, j+0.5):
  self.E1 = 0.5 * (self.E1 + af.shift(self.E1, -1))
  self.E2 = 0.5 * (self.E2 + af.shift(self.E2, 0, -1))

  af.eval(self.E1, self.E2)
  return

def fft_poisson(self):

  if(self._da.getSize()!=1):
    raise Exception('FFT solver can only be used when run in serial')

  else:
    rho  = self.compute_moments('density')
    k_q1 = af.to_array(fftfreq(rho.shape[0], self.dq1))
    k_q2 = af.to_array(fftfreq(rho.shape[1], self.dq2))

    k_q1 = af.tile(k_q1, 1, rho.shape[1])
    k_q2 = af.tile(af.reorder(k_q2), rho.shape[0], 1)

    rho_hat       = af.fft2(rho)
    
    potential_hat       = (1/(4 * np.pi**2 * (k_q1**2 + k_q2**2))) * rho_hat
    potential_hat[0, 0] = 0
    
    E1_hat = -1j * 2 * np.pi * (k_q1) * potential_hat
    E2_hat = -1j * 2 * np.pi * (k_q2) * potential_hat

    self.E1 = af.real(af.ifft2(E1_hat))
    self.E2 = af.real(af.ifft2(E2_hat))

    af.eval(self.E1, self.E2)
    return