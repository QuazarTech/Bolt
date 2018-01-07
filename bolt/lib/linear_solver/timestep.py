#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import arrayfire as af

from . import integrators
from .df_hat_dt import df_hat_dt
from .dfields_hat_dt import dfields_hat_dt


def RK5_step(self, dt):
    self.Y = integrators.RK5(dY_dt, self.Y, dt, self)
    # Solving for tau = 0 systems
    if(    self.single_mode_evolution == False
       and af.any_true(self.physical_system.params.tau(self.q1_center, self.q2_center, 
                                                       self.p1, self.p2, self.p3
                                                      ) == 0
                      )
      ):
        f_hat = self.Y[:, :, :, 0]
        f     = af.real(af.ifft2(0.5 * self.N_q2 * self.N_q1 * f_hat))

        self.Y[:,:, :, 0] = 2 * af.fft2(self._source(f, self.q1_center, self.q2_center,
                                                     self.p1, self.p2, self.p3, 
                                                     self.compute_moments, 
                                                     self.physical_system.params, 
                                                     True
                                                    ) 
                                       )/(self.N_q2 * self.N_q1)
    return

def RK4_step(self, dt):
    self.f_hat, self.fields_hat = integrators.RK4(df_hat_dt, self.f_hat,
                                                  dfields_hat_dt, self.fields_hat,
                                                  dt, self
                                                 )

    # Solving for tau = 0 systems
    # if(    self.single_mode_evolution == False
    #    and af.any_true(self.physical_system.params.tau(self.q1_center, self.q2_center, 
    #                                                    self.p1, self.p2, self.p3
    #                                                   ) == 0
    #                   )
    #   ):
    #     f_hat = self.Y[:, :, :, 0]
    #     f     = af.real(af.ifft2(0.5 * self.N_q2 * self.N_q1 * f_hat))

    #     self.Y[:,:, :, 0] = 2 * af.fft2(self._source(f, self.q1_center, self.q2_center,
    #                                                  self.p1, self.p2, self.p3, 
    #                                                  self.compute_moments, 
    #                                                  self.physical_system.params, 
    #                                                  True
    #                                                 ) 
    #                                    )/(self.N_q2 * self.N_q1)
    return

def RK2_step(self, dt):
    self.Y = integrators.RK2(dY_dt, self.Y, dt, self)
    # Solving for tau = 0 systems
    if(    self.single_mode_evolution == False
       and af.any_true(self.physical_system.params.tau(self.q1_center, self.q2_center, 
                                                       self.p1, self.p2, self.p3
                                                      ) == 0
                      )
      ):
        f_hat = self.Y[:, :, :, 0]
        f     = af.real(af.ifft2(0.5 * self.N_q2 * self.N_q1 * f_hat))

        self.Y[:,:, :, 0] = 2 * af.fft2(self._source(f, self.q1_center, self.q2_center,
                                                     self.p1, self.p2, self.p3, 
                                                     self.compute_moments, 
                                                     self.physical_system.params, 
                                                     True
                                                    ) 
                                       )/(self.N_q2 * self.N_q1)
    return
