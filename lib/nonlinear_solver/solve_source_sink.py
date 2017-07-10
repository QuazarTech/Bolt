#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def RK2(self, dt):
  
  f_initial = self.f.copy() # Storing the value at the start
  self.f    = self.f    + self.g() * (dt/2) # Obtaining value at midpoint(dt/2)
  self.f    = f_initial + self.g() * dt

  af.eval(self.f)
  return

def RK4(self, dt):
  
  f_initial = self.f.copy() # Storing the value at the start

  k1     = self.g()
  self.f = f_initial + 0.5 * k1 * dt
  k2     = self.g()
  self.f = f_initial + 0.5 * k2 * dt
  k3     = self.g()
  self.f = f_initial + k3 * dt
  k4     = self.g()
  
  self.f = f_initial + ((k1+2*k2+2*k3+k4)/6) * dt

  af.eval(self.f)
  return

def RK6(self, dt):
  
  f_initial = self.f.copy() # Storing the value at the start

  k1     = self.g()
  self.f = f_initial + 0.25 * k1 * dt
  k2     = self.g()
  self.f = f_initial + (3/32)*(k1+3*k2)*dt
  k3     = self.g()
  self.f = f_initial + (12/2197)*(161*k1-600*k2+608*k3)*dt
  k4     = self.g()
  self.f = f_initial + (1/4104)*(8341*k1-32832*k2+29440*k3-845*k4)*dt
  k5     = self.g()
  self.f = f_initial + (-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5)*dt
  k6     = self.g()
  
  self.f = f_initial + 1/5*((16/27)*k1+(6656/2565)*k3+(28561/11286)*k4-(9/10)*k5+(2/11)*k6)*dt

  af.eval(self.f)
  return