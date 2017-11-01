#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def calculate_dfdp_background(self):

    """
    Calculates the derivative of the background distribution 
    with respect to the variables p1, p2, p3.
    """
    f_b = self.f_background

    # Using a 4th order central difference stencil:
    self.dfdp1_background = (-np.roll(f_b, -2) + 8 * np.roll(f_b, -1)
                             +np.roll(f_b,  2) - 8 * np.roll(f_b,  1)
                            ) / (12 * self.dp1)

    self.dfdp2_background = (-np.roll(f_b,-2, 1) + 8 * np.roll(f_b, -1, 1)
                             +np.roll(f_b, 2, 1) - 8 * np.roll(f_b,  1, 1)
                            ) / (12 * self.dp2)

    self.dfdp3_background = (-np.roll(f_b, -2, 2) + 8 * np.roll(f_b, -1, 2)
                             +np.roll(f_b,  2, 2) - 8 * np.roll(f_b,  1, 2)
                            ) / (12 * self.dp3)

    return
