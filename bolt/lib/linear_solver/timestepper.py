#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af


def _RK6(derivative_function, vector, dt):
    k1 = derivative_function(vector)
    k2 = derivative_function(vector + 0.25 * k1 * dt)
    k3 = derivative_function(vector + (3 / 32) * (k1 + 3 * k2) * dt)
    k4 = derivative_function(vector+ (12 / 2197) *
                            (161 * k1 - 600 * k2 + 608 * k3) * dt)
    
    k5 = derivative_function(vector + (1 / 4104) * 
                             (8341 * k1 - 32832 * k2 + 
                              29440 * k3 - 845 * k4) * dt)

    k6 = derivative_function(vector + (-(8 / 27) * k1 + 2 * k2 - 
                             (3544 / 2565) * k3 +(1859 / 4104) * k4 
                             - (11 / 40) * k5) * dt)

    vector = vector + 1 / 5 * ((16 / 27) * k1 + (6656 / 2565) * k3 + 
                              (28561 / 11286) * k4 - (9 / 10) * k5 +
                               (2 / 11) * k6) * dt

    af.eval(vector)

    del k1, k2, k3, k4, k5, k6; af.device_gc()

    return(vector)

def _RK4(derivative_function, vector, dt):
    k1 = derivative_function(vector)
    k2 = derivative_function(vector + 0.5 * k1 * dt)
    k3 = derivative_function(vector + 0.5 * k2 * dt)
    k4 = derivative_function(vector + k3 * dt)

    vector = vector + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    af.eval(vector)

    del k1, k2, k3, k4; af.device_gc()

    return(vector)


def _RK2(derivative_function, vector, dt):

    k1 = derivative_function(vector)
    k2 = derivative_function(vector + 0.5 * k1 * dt)

    vector = vector + k2 * dt

    af.eval(vector)

    del k1, k2; af.device_gc()

    return(vector)

def RK6_step(self, dt):
    """
    Evolves the various mode perturbation arrays by a single time-step by
    making use of the RK-6 time-stepping scheme. This scheme is 5th order
    accurate in time.
    """
    self.f_hat = _RK6(self._df_dt, self.f_hat, dt)
    self.Y     = _RK6(self._dY_dt, self.Y, dt)

    af.eval(self.f_hat, self.Y)
    return


def RK4_step(self, dt):
    """
    Evolves the various mode perturbation arrays by a single time-step by
    making use of the RK-4 time-stepping scheme. This scheme is 4th order
    accurate in time.
    """
    self.f_hat = _RK4(self._df_dt, self.f_hat, dt)
    self.Y     = _RK4(self._dY_dt, self.Y, dt)

    af.eval(self.f_hat, self.Y)
    return


def RK2_step(self, dt):
    """
    Evolves the various mode perturbation arrays by a single time-step by
    making use of the RK-2 time-stepping scheme. This scheme is 2nd order
    accurate in time.
    """   
    self.f_hat = _RK2(self._df_dt, self.f_hat, dt)
    self.Y     = _RK2(self._dY_dt, self.Y, dt)
    
    af.eval(self.f_hat, self.Y)
    return
