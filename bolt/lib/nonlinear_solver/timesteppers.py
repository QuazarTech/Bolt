#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def RK2(dx_dt, x_initial, dt, *args):

    # Obtaining value at midpoint(dt/2)
    x = x + dx_dt(x_initial, *args) * (dt / 2)
    x = x_initial + dx_dt(x, *args) * dt

    af.eval(x)
    return

def RK4(dx_dt, x_initial, dt, *args):

    k1 = dx_dt(x_initial, *args)
    x  = x_initial + 0.5 * k1 * dt
    k2 = dx_dt(x, *args)
    x  = x_initial + 0.5 * k2 * dt
    k3 = dx_dt(x, *args)
    x  = x_initial + k3 * dt
    k4 = dx_dt(x, *args)

    x = x_initial + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    af.eval(x)
    return

def RK6(self, dt):

    k1 = dx_dt(x_initial, *args)
    x  = x_initial + 0.25 * k1 * dt
    
    k2 = dx_dt(x, *args)
    x  = x_initial + (3 / 32) * (k1 + 3 * k2) * dt
    
    k3 = dx_dt(x, *args)
    x  = x_initial + (12 / 2197) * (161 * k1 - 600 * k2 + 608 * k3) * dt
    
    k4 = dx_dt(x, *args)
    x  = x_initial + (1 / 4104) * (  8341  * k1 - 32832 * k2
                                   + 29440 * k3 - 845   * k4
                                  ) * dt

    k5 = dx_dt(x, *args)
    x  = x_initial + (- (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3
                      + (1859 / 4104) * k4 - (11 / 40) * k5
                     ) * dt

    k6 = dx_dt(x, *args)
    x  = x_initial + 1 / 5 * (  (16 / 27) * k1 + (6656 / 2565) * k3
                              + (28561 / 11286) * k4 - (9 / 10) * k5
                              + (2 / 11) * k6
                             ) * dt

    af.eval(x)
    return
