"""
This file hold the timestepper function which is to 
be used when the FDTD solver is to be used with the 
finite volume method in p-space.
"""

import arrayfire as af 
from .df_dt_fvm import df_dt_fvm
from bolt.lib.nonlinear_solver.EM_fields_solver.fdtd_explicit \
    import fdtd, fdtd_grid_to_ck_grid

def fvm_timestep_RK2(self, dt):
    
    self._communicate_f()
    self._apply_bcs_f()

    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self, True) * (dt / 2)

    self._communicate_f()
    self._apply_bcs_f()

#    if(    self.physical_system.params.charge_electron != 0
#       and self.physical_system.params.fields_solver == 'fdtd'
#      ):
#        
#        # Will return a flattened array containing the values of
#        # J1,2,3 in 2D space:
#        self.J1 =   self.physical_system.params.charge_electron \
#                  * self.compute_moments('mom_p1_bulk')  # (i + 1/2, j + 1/2)
#        self.J2 =   self.physical_system.params.charge_electron \
#                  * self.compute_moments('mom_p2_bulk')  # (i + 1/2, j + 1/2)
#        self.J3 =   self.physical_system.params.charge_electron \
#                  * self.compute_moments('mom_p3_bulk')  # (i + 1/2, j + 1/2)
#
#        # Obtaining the values for current density on the Yee-Grid:
#        self.J1 = 0.5 * (self.J1 + af.shift(self.J1, 0, 0, 1))  # (i + 1/2, j)
#        self.J2 = 0.5 * (self.J2 + af.shift(self.J2, 0, 1, 0))  # (i, j + 1/2)
#
#        self.J3 = 0.25 * (  self.J3 + af.shift(self.J3, 0, 1, 0) +
#                          + af.shift(self.J3, 0, 0, 1)
#                          + af.shift(self.J3, 0, 1, 1)
#                         )  # (i, j)
#
#        # Here:
#        # cell_centered_EM_fields[:3] is at n
#        # cell_centered_EM_fields[3:] is at n+1/2
#        # cell_centered_EM_fields_at_n_plus_half[3:] is at n-1/2
#
#        self.cell_centered_EM_fields_at_n[:3] = self.cell_centered_EM_fields[:3]
#        self.cell_centered_EM_fields_at_n[3:] = \
#            0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[3:] 
#                   + self.cell_centered_EM_fields[3:]
#                  )
#
#
#        self.cell_centered_EM_fields_at_n_plus_half[3:] = self.cell_centered_EM_fields[3:]
#
#        fdtd(self, dt)
#        fdtd_grid_to_ck_grid(self)
#
#        # Here
#        # cell_centered_EM_fields[:3] is at n+1
#        # cell_centered_EM_fields[3:] is at n+3/2
#
#        self.cell_centered_EM_fields_at_n_plus_half[:3] = \
#            0.5 * (  self.cell_centered_EM_fields_at_n_plus_half[:3] 
#                   + self.cell_centered_EM_fields[:3]
#                  )
    
    self.f = f_initial + df_dt_fvm(self.f, self, False) * dt

    af.eval(self.f)
    return
