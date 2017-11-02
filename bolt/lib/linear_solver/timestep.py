#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from . import integrators
from .dY_dt import dY_dt

def RK6_step(self, dt):
    self.Y = integrators.RK6(dY_dt, self.Y, dt, self)
    return

def RK4_step(self, dt):
    self.Y = integrators.RK4(dY_dt, self.Y, dt, self)
    return

def RK2_step(self, dt):
    self.Y = integrators.RK2(dY_dt, self.Y, dt, self)
    return
