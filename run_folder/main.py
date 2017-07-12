import numpy as np

from lib.physical_system import physical_system
from lib.linear_solver.linear_system import linear_system

import domain
import boundary_conditions
import params
import pylab as pl

from initialize import intial_conditions
import src.nonrelativistic_boltzmann.advection_terms as advection_terms
import src.nonrelativistic_boltzmann.moment_defs as moment_defs

def rhs():
  return 0

system = physical_system(domain, boundary_conditions, intial_conditions, advection_terms, rhs, moment_defs)
ls     = linear_system(system)

ls.init(params)

t  = np.linspace(0, 1, 1000)
dt = t[1] - t[0]

data = np.zeros_like(t)

for time_index, t0 in enumerate(t):
  ls.time_step(dt)
  data[time_index] = np.max(ls.compute_moments('density'))

print(data)
pl.plot(data)
pl.show()
