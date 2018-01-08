"""
This file hold the timestepper function which is to 
be used when the FDTD solver is to be used with the 
finite volume method in p-space.
"""

import arrayfire as af 
from .df_dt_fvm import df_dt_fvm

def fvm_timestep_RK2(self, dt):
    
    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self, True) * (dt / 2)

    self._communicate_f()
    self._apply_bcs_f()

    # Evolving fields:
    if(self.physical_system.params.EM_fields_on == True):
        self.fields_solver.evolve_fields(dt)

    self.f = f_initial + df_dt_fvm(self.f, self, False) * dt

    af.eval(self.f)
    return
