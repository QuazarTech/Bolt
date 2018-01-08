#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def RK2(dx_dt, x_initial, dy_dt, y_initial, dt, *args):

    k1_x = dx_dt(x_initial, *args)
    k1_y = dy_dt(y_initial, *args)

    x = x_initial + k1_x * (dt / 2)
    y = y_initial + k1_y * (dt / 2)

    k2_x = dx_dt(x, *args)
    k2_y = dy_dt(y, *args)

    x = x_initial + k2_x * dt
    y = y_initial + k2_y * dt

    return(x, y)

def RK4(dx_dt, x_initial, dy_dt, y_initial, dt, *args):

    k1_x = dx_dt(x_initial, *args)
    k1_y = dy_dt(y_initial, *args)

    x = x_initial + 0.5 * k1_x * dt
    y = y_initial + 0.5 * k1_y * dt

    k2_x = dx_dt(x, *args)
    k2_y = dy_dt(y, *args)

    x = x_initial + 0.5 * k2_x * dt
    y = y_initial + 0.5 * k2_y * dt

    k3_x = dx_dt(x, *args)
    k3_y = dy_dt(y, *args)

    x = x_initial + k3_x * dt
    y = y_initial + k3_y * dt

    k4_x = dx_dt(x, *args)
    k4_y = dy_dt(y, *args)

    x = x_initial + ((k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6) * dt
    y = y_initial + ((k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6) * dt

    return(x, y)

def RK5(dx_dt, x_initial, dt, *args):

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

    return(x)
