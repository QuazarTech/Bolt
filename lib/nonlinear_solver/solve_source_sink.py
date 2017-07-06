#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import arrayfire as af

def RK2(self, dt):
  
  f_initial = self.f.copy() # Storing the value at the start
  self.f    = self.f    + self.g(self) * (dt/2) # Obtaining value at midpoint(dt/2)
  self.f    = f_initial + self.g(self) * dt

  af.eval(self.f)
  return(self.f)

def RK4(self, dt):
  
  f_initial = self.f.copy() # Storing the value at the start

  k1     = self.g(self)
  self.f = f_initial + 0.5 * k1 * dt
  k2     = self.g(self)
  self.f = f_initial + 0.5 * k2 * dt
  k3     = self.g(self)
  self.f = f_initial + k3 * dt
  k4     = self.g(self)
  
  self.f = f_initial + ((k1+2*k2+2*k3+k4)/6) * dt

  af.eval(self.f)
  return(self.f)