"""
This file hold the timestepper function which is to 
be used when the FDTD solver is to be used with the 
finite volume method in p-space.
"""

import arrayfire as af 
from .df_dt_fvm import df_dt_fvm
from bolt.lib.nonlinear.utils.broadcasted_primitive_operations import multiply

def fvm_timestep_RK2(self, dt):
    
    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self) * (dt / 2)

    self._communicate_f()
    self._apply_bcs_f()

    # Evolving fields:
    if(self.physical_system.params.fields_solver == 'fdtd'):
        
        J1 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v1_bulk')
                     )  # (i + 1/2, j + 1/2)
        J2 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v2_bulk')
                     )  # (i + 1/2, j + 1/2)
        J3 = multiply(self.physical_system.params.charge, 
                      self.compute_moments('mom_v3_bulk')
                     )  # (i + 1/2, j + 1/2)

        self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

    self.f = f_initial + df_dt_fvm(self.f, self) * dt

    af.eval(self.f)
    return
