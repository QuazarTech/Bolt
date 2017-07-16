#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import arrayfire as af

# Importing solver functions:
from lib.nonlinear_solver.interpolation_routines import f_interp_2d
from lib.nonlinear_solver.solve_source_sink import RK2, RK4, RK6
from lib.nonlinear_solver.EM_fields_solver.fields_step import fields_step

def strang_timestep(self, dt):
  if(self.physical_system.params.timestepper == 'RK2'):
    solve_source_sink = RK2
  elif(self.physical_system.params.timestepper == 'RK4'):
    solve_source_sink = RK4
  elif(self.physical_system.params.timestepper == 'RK6'):
    solve_source_sink = RK6
  else:
    raise NotImplementedError('Timestepper option invalid/not implemented')

  # Advection in position space:
  f_interp_2d(self, 0.25*dt)
  self._communicate_distribution_function()
  # Solving the source/sink terms:
  solve_source_sink(self, 0.5*dt)
  self._communicate_distribution_function()
  # Advection in position space:
  f_interp_2d(self, 0.25*dt)
  self._communicate_distribution_function()

  # Advection in velocity space:
  fields_step(self, dt)
  self._communicate_distribution_function()

  # Advection in position space:
  f_interp_2d(self, 0.25*dt)
  self._communicate_distribution_function()
  # Solving the source/sink terms:
  solve_source_sink(self, 0.5*dt)
  self._communicate_distribution_function()
  # Advection in position space:
  f_interp_2d(self, 0.25*dt)
  self._communicate_distribution_function()

  af.eval(self.f)
  return(self.f)

def lie_timestep(self, dt):
  if(self.physical_system.params.timestepper == 'RK2'):
    solve_source_sink = RK2
  elif(self.physical_system.params.timestepper == 'RK4'):
    solve_source_sink = RK4
  elif(self.physical_system.params.timestepper == 'RK6'):
    solve_source_sink = RK6
  else:
    raise NotImplementedError('Timestepper option invalid/not implemented')

  # Advection in position space:
  f_interp_2d(self, dt)

  # Solving the source/sink terms:
  solve_source_sink(self, dt)
  self._communicate_distribution_function()

  # Advection in velocity space:
  fields_step(self, dt)
  self._communicate_distribution_function()

  af.eval(self.f)
  return(self.f)

def time_step(self, dt):
  if(self.physical_system.params.time_splitting == 'strang'):
    return(strang_timestep)
  elif(self.physical_system.params.time_splitting == 'lie'):
    return(lie_timestep)
  else:
    raise NotImplementedError('Time-splitting method invalid/not-implemented')