import numpy as np

from lib.physical_system import physical_system
from lib.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
from f_initial import f_maxwell_boltzmann
import advection_terms

def rhs():
  return 0

system           = physical_system(domain, boundary_conditions, f_maxwell_boltzmann, advection_terms, rhs)
nonlinear_system = nonlinear_solver(physical_system)

time = np.linspace(0, 1, 100)

system.evolve(time)