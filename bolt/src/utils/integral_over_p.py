#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import arrayfire as af

def integral_over_p(array):
    return(af.sum(array, 0))
