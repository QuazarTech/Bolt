#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_v import integral_over_v

def density(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f, integral_measure))

def mom_v1_bulk(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v1, integral_measure))

def mom_v2_bulk(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v2, integral_measure))

def mom_v3_bulk(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v3, integral_measure))

def energy(f, v1, v2, v3, integral_measure):
    return(integral_over_v(0.5 * f * (v1**2 + v2**2 + v3**2),
                           integral_measure
                          )
          )
