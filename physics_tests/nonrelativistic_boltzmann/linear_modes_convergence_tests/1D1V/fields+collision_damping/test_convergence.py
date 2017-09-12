import numpy as np
import arrayfire as af
from physics_tests.nonrelativistic_boltzmann.run_cases import run_cases
from physics_tests.nonrelativistic_boltzmann.check_convergence import check_convergence

def tau(q1, q2, p1, p2, p3):
    return(af.constant(np.inf, q1.shape[0], q2.shape[1],
                       p1.shape[2], dtype = af.Dtype.f64
                      )
          )

def test_convergence():
    run_cases(-10, tau)
    check_convergence()
