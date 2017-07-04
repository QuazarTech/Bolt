import arrayfire as af

def f_interp_2d(self, dt):
  # Since the interpolation function are being performed in position space,
  # the arrays used in the computation need to be in positionsExpanded form.

  # Obtaining the left-bottom corner coordinates
  # (lowest values of the canonical coordinates in the local zone)
  # Additionally, we also obtain the size of the local zone
  ((i_q1_lowest, i_q2_lowest), (N_q1_local, N_q2_local)) = self.physical_system.da.getCorners()
  
  q1_center_new = self.physical_system.q1_center - self.physical_system.A_q1*dt
  q2_center_new = self.physical_system.q2_center - self.physical_system.A_q2*dt

  # Obtaining the center coordinates:
  (i_q1_center, i_q2_center) = (i_q1_lowest + 0.5, i_q2_lowest + 0.5)

  # Obtaining the left, and bottom boundaries for the local zones:
  q1_boundary_lower = self.physical_system.q1_start + i_q1_center * self.physical_system.dq1
  q2_boundary_lower = self.physical_system.q2_start + i_q2_center * self.physical_system.dq2

  # Adding N_ghost to account for the offset due to ghost zones:
  q1_interpolant = (q1_center_new - q1_boundary_lower)/self.physical_system.dq1 + self.physical_system.N_ghost
  q2_interpolant = (q2_center_new - q2_boundary_lower)/self.physical_system.dq2 + self.physical_system.N_ghost

  self.log_f = af.approx2(self.log_f,\
                          q1_interpolant,\
                          q2_interpolant,\
                          af.INTERP.BICUBIC_SPLINE
                         )

  af.eval(self.log_f)
  return(self.log_f)
