import numpy as np
import arrayfire as af

from physics_tests.nonrelativistic_boltzmann.run_cases import run_cases
from physics_tests.nonrelativistic_boltzmann.check_convergence import check_convergence

@af.broadcast
def tau_collisional(q1, q2, p1, p2, p3):
    return(0.01 * q1**0 * p1**0)

@af.broadcast
def tau_collisionless(q1, q2, p1, p2, p3):
    return(np.inf * q1**0 * p1**0)

def test_collisionless():
    run_cases(1, 2, 0, tau_collisionless)
    check_convergence()

def test_collisional():
    run_cases(1, 2, 0, tau_collisional)
    check_convergence()

def test_fields_collisionless():
    run_cases(1, 2, -10, tau_collisionless)
    check_convergence()

def test_fields_collisional():
    run_cases(1, 2, -10, tau_collisional)
    check_convergence()
