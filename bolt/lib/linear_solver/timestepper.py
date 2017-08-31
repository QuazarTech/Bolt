#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import arrayfire as af

def RK6_step(self, dt):
    """
    Evolves the various modes by a single time-step by
    making use of the RK-6 time-stepping scheme. 
    This scheme is 5th order accurate in time.
    """
    k1 = self._dY_dt(self.Y)
    k2 = self._dY_dt(self.Y + 0.25*k1*dt)
    k3 = self._dY_dt(self.Y + (3/32)*(k1+3*k2)*dt)
    k4 = self._dY_dt(self.Y + (12/2197)*(161*k1-600*k2+608*k3)*dt)
    k5 = self._dY_dt(self.Y + (1/4104)*(8341*k1-32832*k2+29440*k3-845*k4)*dt)
    k6 = self._dY_dt(self.Y + (-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5)*dt)

    self.Y = self.Y + 1/5*((16/27)*k1+(6656/2565)*k3+(28561/11286)*k4-(9/10)*k5+(2/11)*k6)*dt

    af.eval(self.Y)
    return

def RK4_step(self, dt):
    """
    Evolves the various modesby a single time-step by
    making use of the RK-4 time-stepping scheme. 
    This scheme is 4th order accurate in time.
    """
    k1 = self._dY_dt(self.Y)
    k2 = self._dY_dt(self.Y + 0.5*k1*dt)
    k3 = self._dY_dt(self.Y + 0.5*k2*dt)
    k4 = self._dY_dt(self.Y + k3*dt)

    self.Y = self.Y + ((k1+2*k2+2*k3+k4)/6)*dt

    af.eval(self.Y)
    return

def RK2_step(self, dt):
    """
    Evolves the various modes by a single time-step by
    making use of the RK-2 time-stepping scheme. 
    This scheme is 2nd order accurate in time.
    """
    k1 = self._dY_dt(self.Y)
    k2 = self._dY_dt(self.Y + 0.5*k1*dt)

    self.Y = self.Y + k2*dt

    af.eval(self.Y)
    return
