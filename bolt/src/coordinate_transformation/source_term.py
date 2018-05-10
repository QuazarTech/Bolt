"""Contains the function which returns the Source/Sink term."""


import numpy as np
import arrayfire as af

@af.broadcast
def source_term(f, t, r, theta, rdot, thetadot, phidot, moments, params, flag = False):
    """Return BGK operator -(f-f0)/tau."""
    return(-2 * rdot * f / r)
