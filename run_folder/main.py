# import numpy as np

from lib.physical_system import physical_system
from lib.linear_solver.linear_system import linear_system

import domain
import boundary_conditions
import params
from initialize import intial_conditions
import src.nonrelativistic_boltzmann.advection_terms as advection_terms
import src.nonrelativistic_boltzmann.moment_defs as moment_defs

def rhs():
  return 0

system = physical_system(domain, boundary_conditions, intial_conditions, advection_terms, rhs, moment_defs)
ls     = linear_system(system)

ls.init(params)

import pylab as pl
print(ls.compute_moments('density'))
pl.plot(ls.compute_moments('E'))
pl.show()
# time = np.linspace(0, 1, 100)

# linear_system.evolve(time)