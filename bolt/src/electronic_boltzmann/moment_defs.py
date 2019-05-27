#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_v import integral_over_v

#TODO : Import required constants from libraries
#     : pi = 3.1415926535
#     : h_bar = 1.0545718e-4

def density(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f, integral_measure))

def j_x(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * (4./(2.*3.1415926535*1.0545718e-4)**2) * v1, integral_measure))

def j_y(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * (4./(2.*3.1415926535*1.0545718e-4)**2) * v2, integral_measure))
