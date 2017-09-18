import numpy as np
import arrayfire as af

from physics_tests.nonrelativistic_boltzmann.run_cases import run_cases
from physics_tests.nonrelativistic_boltzmann.check_convergence import check_convergence

def tau_collisional(q1, q2, p1, p2, p3):
    return(af.constant(0.01, q1.shape[0], q2.shape[1],
                       p1.shape[2], dtype = af.Dtype.f64
                      )
          )

def tau_collisionless(q1, q2, p1, p2, p3):
    return(af.constant(np.inf, q1.shape[0], q2.shape[1],
                       p1.shape[2], dtype = af.Dtype.f64
                      )
          )

def test_collisionless():
    run_cases(1, 1, 0, tau_collisionless)
    check_convergence()

def test_collisional():
    run_cases(1, 1, 0, tau_collisional)
    check_convergence()

def test_fields_collisionless():
    run_cases(1, 1, -10, tau_collisionless)
    check_convergence()

def test_fields_collisional():
    run_cases(1, 1, -10, tau_collisional)
    check_convergence()
