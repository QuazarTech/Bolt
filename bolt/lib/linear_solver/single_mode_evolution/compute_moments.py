#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def compute_moments(self, moment_name):
    moment_exponents = \
            np.array(
                     self.physical_system.moment_exponents[moment_name]
                    )

    moment_coeffs = \
            np.array(
                     self.physical_system.moment_coeffs[moment_name]
                    )

    # This checks that the moment definition is of the form
    # [[a, b, c], [d, e, f]...]. Alternatively if it isn't it checks
    # whether the definition is of the form [a, b, c]
    try:
        moment_variable = 1
        for i in range(moment_exponents.shape[0]):
            moment_variable *=   moment_coeffs[i, 0] \
                               * self.p1**(moment_exponents[i, 0]) \
                               + moment_coeffs[i, 1] \
                               * self.p2**(moment_exponents[i, 1]) \
                               + moment_coeffs[i, 2] \
                               * self.p3**(moment_exponents[i, 2])
    except BaseException:
        moment_variable =   moment_coeffs[0] * self.p1**(moment_exponents[0]) \
                          + moment_coeffs[1] * self.p2**(moment_exponents[1]) \
                          + moment_coeffs[2] * self.p3**(moment_exponents[2])

    # af.broadcast(function, *args) performs batched operations on
    # function(*args):
    moment_hat = np.sum(self.Y[0] * moment_variable) * self.dp3 * self.dp2 * self.dp1
    return(moment_hat)
