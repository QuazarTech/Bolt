#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import arrayfire as af

def integral_over_v(array, integral_measure):
    return(af.sum(array, 0) * integral_measure)
