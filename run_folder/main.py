import numpy as np

from lib.physical_system import physical_system
from lib.linear_solver import linear_solver

import domain
import boundary_conditions
from f_initial import f_maxwell_boltzmann
import src.nonrelativistic_boltzmann.advection_terms as advection_terms

def rhs():
  return 0

system        = physical_system(domain, boundary_conditions, f_maxwell_boltzmann, advection_terms, rhs)
linear_system = linear_solver(physical_system)

time = np.linspace(0, 1, 100)

linear_system.evolve(time)