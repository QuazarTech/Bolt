import arrayfire as af
import numpy as np

from bolt.lib.physical_system import physical_system
from bolt.lib.linear.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

# Declaring the solver object which will evolve the defined physical system:
ls  = linear_solver(system)

n    = ls.compute_moments('density')
v1_b = ls.compute_moments('mom_v1_bulk') / n
E    = ls.compute_moments('energy')

T = (2 * E - n * v1_b**2) / n

assert(af.mean(af.abs(n[0, 0] - (1   + 0.01 * af.cos(2 * np.pi * ls.q1_center))))<1e-13)
assert(af.mean(af.abs(n[0, 1] - (1   + 0.02 * af.cos(4 * np.pi * ls.q1_center))))<1e-13)
assert(af.mean(af.abs(T[0, 0] - (1   + 0.02 * af.sin(2 * np.pi * ls.q1_center))))<1e-13)
assert(af.mean(af.abs(T[0, 1] - (100 + 0.01 * af.sin(4 * np.pi * ls.q1_center))))<1e-13)
