import arrayfire as af
import numpy as np
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver

from bolt.lib.nonlinear_solver.tests.performance.input_files \
    import domain
from bolt.lib.nonlinear_solver.tests.performance.input_files \
    import boundary_conditions
from bolt.lib.nonlinear_solver.tests.performance.input_files \
    import params
from bolt.lib.nonlinear_solver.tests.performance.input_files \
    import initialize


import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator \
    as collision_operator
import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs