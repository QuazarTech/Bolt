import numpy as np
import arrayfire as af

from physics_tests.nonrelativistic_boltzmann.run_cases_nls import run_cases

@af.broadcast
def tau_collisional(q1, q2, p1, p2, p3):
    return(0.01 * q1**0 * p1**0)

@af.broadcast
def tau_collisionless(q1, q2, p1, p2, p3):
    return(np.inf * q1**0 * p1**0)

def run_collisionless():
    run_cases(2, 2, 0, tau_collisionless)

def run_collisional():
    run_cases(2, 2, 0, tau_collisional)

def run_fields_collisionless():
    run_cases(2, 2, -10, tau_collisionless)

def run_fields_collisional():
    run_cases(2, 2, -10, tau_collisional)
