#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def calculate_dfdp_background(self):
    """
    Calculates the derivative of the background distribution 
    with respect to the variables p1, p2, p3. This is used to
    solve for the contribution from the fields
    """
    f_b = af.moddims(self.f_background, self.N_p1, self.N_p2, self.N_p3, self.N_species)

#    8th order central diff -- just in case
#    f_minus_4 = af.shift(f_b, 4); f_plus_4 = af.shift(f_b, -4)
#    f_minus_3 = af.shift(f_b, 3); f_plus_3 = af.shift(f_b, -3)
#    f_minus_2 = af.shift(f_b, 2); f_plus_2 = af.shift(f_b, -2)
#    f_minus_1 = af.shift(f_b, 1); f_plus_1 = af.shift(f_b, -1)
#
#    dfdp1_background = \
#    ( 1/280)*f_minus_4 + (-4/105)*f_minus_3 + ( 1/5)*f_minus_2 + (-4/5)*f_minus_1 \
#  + (-1/280)*f_plus_4  + ( 4/105)*f_plus_3  + (-1/5)*f_plus_2  + ( 4/5)*f_plus_1 
#
#    dfdp1_background = dfdp1_background/self.dp1

    # Using a 4th order central difference stencil:
    dfdp1_background = (-af.shift(f_b, -2) + 8 * af.shift(f_b, -1)
                        +af.shift(f_b,  2) - 8 * af.shift(f_b,  1)
                       ) / (12 * self.dp1)

    dfdp2_background = (-af.shift(f_b, 0, -2) + 8 * af.shift(f_b, 0, -1)
                        +af.shift(f_b, 0,  2) - 8 * af.shift(f_b, 0,  1)
                       ) / (12 * self.dp2)

    dfdp3_background = (-af.shift(f_b, 0, 0, -2) + 8 * af.shift(f_b, 0, 0, -1)
                        +af.shift(f_b, 0, 0,  2) - 8 * af.shift(f_b, 0, 0,  1)
                       ) / (12 * self.dp3)

    # Reshaping such that the variations in velocity are along axis 0:
    self.dfdp1_background = af.moddims(dfdp1_background, 
                                       self.N_p1 * self.N_p2 * self.N_p3,
                                       self.N_species
                                      )
    self.dfdp2_background = af.moddims(dfdp2_background,
                                       self.N_p1 * self.N_p2 * self.N_p3,
                                       self.N_species
                                      )
    self.dfdp3_background = af.moddims(dfdp3_background,
                                       self.N_p1 * self.N_p2 * self.N_p3,
                                       self.N_species
                                      )

    af.eval(self.dfdp1_background,
            self.dfdp2_background,
            self.dfdp3_background
           )

    return
