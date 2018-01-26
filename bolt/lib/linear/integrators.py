#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def RK2(dx_dt, x_initial, dt, *args):
    """
    Integrates x from x_initial(t = t0) to x(t = t0 + dt) by taking 
    slope from dx_dt, and integrates it using the RK2 method. This method 
    is second order accurate.
    
    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """
    x = x_initial + dx_dt(x_initial, *args) * (dt / 2)
    args[1].time_elapsed += 0.5 * dt 
    x = x_initial + dx_dt(x, *args) * dt
    args[1].time_elapsed += 0.5 * dt 

    return(x)

def RK4(dx_dt, x_initial, dt, *args):
    """
    Integrates x from x_initial(t = t0) to x(t = t0 + dt) by taking 
    slope from dx_dt, and integrates it using the RK4 method. This method 
    is fourth order accurate.
    
    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """
    k1 = dx_dt(x_initial, *args)
    x  = x_initial + 0.5 * k1 * dt

    args[1].time_elapsed += 0.5 * dt 

    k2 = dx_dt(x, *args)
    x  = x_initial + 0.5 * k2 * dt

    k3 = dx_dt(x, *args)
    x  = x_initial + k3 * dt

    args[1].time_elapsed += 0.5 * dt 
    
    k4 = dx_dt(x, *args)
    x = x_initial + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    return(x)

def RK5(dx_dt, x_initial, dt, *args):
    """
    Integrates x from x_initial(t = t0) to x(t = t0 + dt) by taking 
    slope from dx_dt, and integrates it using the RK5 method. This method 
    is fifth order accurate.
    
    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """

    k1 = dx_dt(x_initial, *args)
    x  = x_initial + 0.25 * k1 * dt
    
    args[1].time_elapsed += 0.25 * dt 
    
    k2 = dx_dt(x, *args)
    x  = x_initial + (1 / 8) * (k1 + k2) * dt
    
    k3 = dx_dt(x, *args)
    x  = x_initial - (0.5 * k2 - k3) * dt
    
    args[1].time_elapsed += 0.25 * dt 
    
    k4 = dx_dt(x, *args)
    x  = x_initial + (3 * k1 + 9 * k4) * dt/16
    
    args[1].time_elapsed += 0.25 * dt 

    k5 = dx_dt(x, *args)
    x  = x_initial - (3 * k1 - 2 * k2 - 12 * k3 + 12 * k4 - 8 * k5) * dt/7

    args[1].time_elapsed += 0.25 * dt 

    k6 = dx_dt(x, *args)
    x  = x_initial + (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6) * dt/90

    return(x)

# The following coupled integrators are used when 2 vectors which are coupled 
# to each other need to be evolved . These coupled integrators
# ensure that the variables evolved together following same time staggering:
def RK2_coupled(dx_dt, x_initial, dy_dt, y_initial, dt, *args):
    """
    Integrates:
    x from x_initial(t = t0) --> x(t = t0 + dt) 
    y from y_initial(t = t0) --> y(t = t0 + dt)

    by taking slopes from dx_dt and dy_dt, and integrates it using the
    RK2 method. This method is second order accurate.
    
    This method ensures that the vectors evolved are evaluated at the same time:
    To elaborate: if x(t) = x(y(t), t), then when the slope for x ie.dx_dt is 
    evaluated at t1 then x(y(t1), t1) is used to calculate the slope.

    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dy_dt: function
           Returns the slope dy_dt which is 
           used to evolve y

    y_initial: array
               The value of y at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """
    k1_x = dx_dt(x_initial, y_initial, *args)
    k1_y = dy_dt(x_initial, y_initial, *args)

    x = x_initial + k1_x * (dt / 2)
    y = y_initial + k1_y * (dt / 2)

    args[0].time_elapsed += 0.5 * dt 

    k2_x = dx_dt(x, y, *args)
    k2_y = dy_dt(x, y, *args)

    x = x_initial + k2_x * dt
    y = y_initial + k2_y * dt

    args[0].time_elapsed += 0.5 * dt 

    return(x, y)

def RK4_coupled(dx_dt, x_initial, dy_dt, y_initial, dt, *args):
    """
    Integrates:
    x from x_initial(t = t0) --> x(t = t0 + dt) 
    y from y_initial(t = t0) --> y(t = t0 + dt)

    by taking slopes from dx_dt and dy_dt, and integrates it using the
    RK4 method. This method is fourth order accurate.
    
    This method ensures that the vectors evolved are evaluated at the same time:
    To elaborate: if x(t) = x(y(t), t), then when the slope for x ie.dx_dt is 
    evaluated at t1 then x(y(t1), t1) is used to calculate the slope.

    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dy_dt: function
           Returns the slope dy_dt which is 
           used to evolve y

    y_initial: array
               The value of y at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """
    k1_x = dx_dt(x_initial, y_initial, *args)
    k1_y = dy_dt(x_initial, y_initial, *args)

    args[0].time_elapsed += 0.5 * dt 

    x = x_initial + 0.5 * k1_x * dt
    y = y_initial + 0.5 * k1_y * dt

    k2_x = dx_dt(x, y, *args)
    k2_y = dy_dt(x, y, *args)

    x = x_initial + 0.5 * k2_x * dt
    y = y_initial + 0.5 * k2_y * dt

    k3_x = dx_dt(x, y, *args)
    k3_y = dy_dt(x, y, *args)

    x = x_initial + k3_x * dt
    y = y_initial + k3_y * dt

    args[0].time_elapsed += 0.5 * dt 

    k4_x = dx_dt(x, y, *args)
    k4_y = dy_dt(x, y, *args)

    x = x_initial + ((k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6) * dt
    y = y_initial + ((k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6) * dt

    return(x, y)

def RK5_coupled(dx_dt, x_initial, dy_dt, y_initial, dt, *args):
    """
    Integrates:
    x from x_initial(t = t0) --> x(t = t0 + dt) 
    y from y_initial(t = t0) --> y(t = t0 + dt)

    by taking slopes from dx_dt and dy_dt, and integrates it using the
    RK5 method. This method is fifth order accurate.
    
    This method ensures that the vectors evolved are evaluated at the same time:
    To elaborate: if x(t) = x(y(t), t), then when the slope for x ie.dx_dt is 
    evaluated at t1 then x(y(t1), t1) is used to calculate the slope.

    Parameters
    ----------
    
    dx_dt: function
           Returns the slope dx_dt which is 
           used to evolve x

    x_initial: array
               The value of x at the beginning of the
               timestep.

    dy_dt: function
           Returns the slope dy_dt which is 
           used to evolve y

    y_initial: array
               The value of y at the beginning of the
               timestep.

    dt: double
        The timestep size.
    """

    k1_x = dx_dt(x_initial, y_initial, *args)
    k1_y = dy_dt(x_initial, y_initial, *args)

    x = x_initial + 0.25 * k1_x * dt
    y = y_initial + 0.25 * k1_y * dt
    
    args[0].time_elapsed += 0.25 * dt 

    k2_x = dx_dt(x, y, *args)
    k2_y = dy_dt(x, y, *args)

    x = x_initial + (1 / 8) * (k1_x + k2_x) * dt
    x = x_initial + (1 / 8) * (k1_y + k2_y) * dt
    
    k3_x = dx_dt(x, y, *args)
    k3_y = dy_dt(x, y, *args)

    x = x_initial - (0.5 * k2_x - k3_x) * dt
    y = y_initial - (0.5 * k2_y - k3_y) * dt

    args[0].time_elapsed += 0.25 * dt 

    k4_x = dx_dt(x, y, *args)
    k4_y = dy_dt(x, y, *args)

    x  = x_initial + (3 * k1_x + 9 * k4_x) * dt/16
    y  = y_initial + (3 * k1_y + 9 * k4_y) * dt/16

    args[0].time_elapsed += 0.25 * dt 

    k5_x = dx_dt(x, y, *args)
    k5_y = dy_dt(x, y, *args)
    
    x  = x_initial - (3 * k1_x - 2 * k2_x - 12 * k3_x + 12 * k4_x - 8 * k5_x) * dt/7
    y  = y_initial - (3 * k1_y - 2 * k2_y - 12 * k3_y + 12 * k4_y - 8 * k5_y) * dt/7

    args[0].time_elapsed += 0.25 * dt 

    k6_x = dx_dt(x, y, *args)
    k6_y = dy_dt(x, y, *args)

    x = x_initial + (7 * k1_x + 32 * k3_x + 12 * k4_x + 32 * k5_x + 7 * k6_x) * dt/90
    y = y_initial + (7 * k1_y + 32 * k3_y + 12 * k4_y + 32 * k5_y + 7 * k6_y) * dt/90

    return(x, y)
