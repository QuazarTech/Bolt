#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np 
import arrayfire as af
from petsc4py import PETSc

from lib.nonlinear_solver.EM_fields_solver.fdtd_explicit import fdtd
from lib.nonlinear_solver.communicate import communicate_fields

def gauss1D(x, spread):
  return af.exp(-((x - 0.5)**2 )/(2*spread**2))

class test(object):
  def __init__(self, N):
    self.q1_start = 0
    self.q2_start = 0

    self.q1_end = 1
    self.q2_end = 1

    self.N_q1 = N
    self.N_q2 = N

    self.dq1 = (self.q1_end - self.q1_start)/self.N_q1
    self.dq2 = (self.q2_end - self.q2_start)/self.N_q2

    self.N_ghost = np.random.randint(3, 5)

    self.q1 = self.q1_start + (0.5 + np.arange(-self.N_ghost, self.N_q1 + self.N_ghost)) * self.dq1
    self.q2 = self.q2_start + (0.5 + np.arange(-self.N_ghost, self.N_q2 + self.N_ghost)) * self.dq2

    self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
    self.q2, self.q1 = af.to_array(self.q2), af.to_array(self.q1)

    self.E1 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    self.E2 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    self.E3 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    
    self.B1 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    self.B2 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    self.B3 = af.constant(0, self.q1.shape[0], self.q1.shape[1], dtype = af.Dtype.f64)
    
    self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],\
                                          dof = 6,\
                                          stencil_width = self.N_ghost,\
                                          boundary_type = ('periodic', 'periodic'),\
                                          stencil_type = 1, \
                                         )

    self._glob_fields  = self._da_fields.createGlobalVec()
    self._local_fields = self._da_fields.createLocalVec()
    
  _communicate_fields = communicate_fields

def test_fdtd_mode1():

  error_B1 = np.zeros(5)
  error_B2 = np.zeros(5)
  error_E3 = np.zeros(5)
  
  N = 2**np.arange(5, 10)
  
  for i in range(N.size):   

    obj = test(N[i])

    N_g = obj.N_ghost

    obj.B1[N_g:-N_g, N_g:-N_g] =\
    gauss1D(obj.q2[N_g:-N_g, N_g:-N_g], 0.1)

    obj.B2[N_g:-N_g, N_g:-N_g] =\
    gauss1D(obj.q1[N_g:-N_g, N_g:-N_g], 0.1)

    dt   = obj.dq1/2
    time = np.arange(dt, 1 + dt, dt)

    E3_initial = obj.E3.copy()
    B1_initial = obj.B1.copy()
    B2_initial = obj.B2.copy()

    obj.J1, obj.J2, obj.J3 = 0, 0, 0

    for time_index, t0 in enumerate(time):
      fdtd(obj, dt)

    error_B1[i] = af.sum(af.abs(obj.B1[N_g:-N_g, N_g:-N_g] -\
                                B1_initial[N_g:-N_g, N_g:-N_g]))/\
                                (B1_initial.elements())

    error_B2[i] = af.sum(af.abs(obj.B2[N_g:-N_g, N_g:-N_g] -\
                                B2_initial[N_g:-N_g, N_g:-N_g]))/\
                                (B2_initial.elements())

    error_E3[i] = af.sum(af.abs(obj.E3[N_g:-N_g, N_g:-N_g] -\
                                E3_initial[N_g:-N_g, N_g:-N_g]))/\
                                (E3_initial.elements())

  print(error_B1, error_B2, error_E3)

  poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
  poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
  poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

  print(poly_B1)
  print(poly_B2)
  print(poly_E3)
  
test_fdtd_mode1()