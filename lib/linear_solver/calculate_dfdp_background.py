#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def calculate_dfdp_background(self):
  """
  Calculates the derivative of the background distribution with respect
  to the variables p1, p2, p3.
  """

  f_b = af.moddims(self.f_background, self.N_p1, self.N_p2, self.N_p3)

  if(self.p_dim == 1):
    dfdp1_background = (-af.shift(f_b, -2)  + 8*af.shift(f_b, -1) +\
                         af.shift(f_b,  2)  - 8*af.shift(f_b,  1) 
                       )/(12*self.dp1) 
    dfdp2_background = af.constant(0, f_b.shape[0], 1, 1, dtype = af.Dtype.f64)
    dfdp3_background = af.constant(0, f_b.shape[0], 1, 1, dtype = af.Dtype.f64)

  elif(self.p_dim == 2):
    dfdp1_background = (-af.shift(f_b, -2)  + 8*af.shift(f_b, -1) +\
                         af.shift(f_b,  2)  - 8*af.shift(f_b,  1) 
                       )/(12*self.dp1) 
    dfdp2_background = (-af.shift(f_b, 0, -2)  + 8*af.shift(f_b, 0, -1) +\
                         af.shift(f_b, 0, 2)   - 8*af.shift(f_b, 0,  1)
                       )/(12*self.dp2)
    dfdp3_background = af.constant(0, f_b.shape[0], f_b.shape[1], 1, dtype = af.Dtype.f64)
  
  else:
    dfdp1_background = (-af.shift(f_b, -2)  + 8*af.shift(f_b, -1) +\
                         af.shift(f_b,  2)  - 8*af.shift(f_b,  1) 
                       )/(12*self.dp1) 
    dfdp2_background = (-af.shift(f_b, 0, -2) + 8*af.shift(f_b, 0, -1) +\
                         af.shift(f_b, 0,  2) - 8*af.shift(f_b, 0,  1)
                       )/(12*self.dp2)
    dfdp3_background = (-af.shift(f_b, 0, 0, -2) + 8*af.shift(f_b, 0, 0, -1) +\
                         af.shift(f_b, 0, 0,  2) - 8*af.shift(f_b, 0, 0,  1)
                       )/(12*self.dp3)

  self.dfdp1_background = af.reorder(af.flat(dfdp1_background), 2, 3, 0, 1)
  self.dfdp2_background = af.reorder(af.flat(dfdp2_background), 2, 3, 0, 1)
  self.dfdp3_background = af.reorder(af.flat(dfdp3_background), 2, 3, 0, 1)

  # positionsExpanded form
  # (N_q1, N_q2, N_p1*N_p2*N_p3)
  self.dfdp1_background = af.tile(self.dfdp1_background, self.N_q1, self.N_q2, 1)
  self.dfdp2_background = af.tile(self.dfdp2_background, self.N_q1, self.N_q2, 1)
  self.dfdp3_background = af.tile(self.dfdp3_background, self.N_q1, self.N_q2, 1)

  af.eval(self.dfdp1_background, self.dfdp2_background, self.dfdp3_background)
  return