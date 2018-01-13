#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import arrayfire as af

from . import integrators
from .df_hat_dt import df_hat_dt
from .fields.dfields_hat_dt import dfields_hat_dt

def RK5_step(self, dt):

    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK5_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )

    else:
        self.f_hat = integrators.RK5(df_hat_dt, self.f_hat,
                                     dt, self
                                    )

    return

def RK4_step(self, dt):
    
    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK4_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )

    else:
        self.f_hat = integrators.RK4(df_hat_dt, self.f_hat,
                                     dt, self
                                    )

    return

def RK2_step(self, dt):
    
    if(    self.physical_system.params.EM_fields_enabled == True 
       and self.physical_system.params.fields_type == 'electrodynamic'
      ):
        self.f_hat, self.fields_solver.fields_hat = \
            integrators.RK2_coupled(df_hat_dt, self.f_hat,
                                    dfields_hat_dt, self.fields_solver.fields_hat,
                                    dt, self
                                   )

    else:
        self.f_hat = integrators.RK2(df_hat_dt, self.f_hat,
                                     dt, self
                                    )

    return
