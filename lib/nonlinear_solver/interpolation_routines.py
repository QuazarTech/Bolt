#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def f_interp_2d(self, dt):
  # Since the interpolation function are being performed in position space,
  # the arrays used in the computation need to be in positionsExpanded form.

  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self._da.getCorners()
  
  q1_center_new = self.q1_center - self.A_q1*dt
  q2_center_new = self.q2_center - self.A_q2*dt

  # Obtaining the center coordinates:
  (i_q1_center, i_q2_center) = (i_q1_lowest + 0.5, i_q2_lowest + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  q1_boundary_lower = self.q1_start + i_q1_center * self.dq1
  q2_boundary_lower = self.q2_start + i_q2_center * self.dq2

  # Adding N_ghost to account for the offset due to ghost zones:
  q1_interpolant = (q1_center_new - q1_boundary_lower)/self.dq1 + self.N_ghost
  q2_interpolant = (q2_center_new - q2_boundary_lower)/self.dq2 + self.N_ghost

  self.f = af.approx2(self.f,\
                      q1_interpolant,\
                      q2_interpolant,\
                      af.INTERP.BICUBIC_SPLINE
                     )

  af.eval(self.f)
  return(self.f)