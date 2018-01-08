#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
from .interpolation_routines import f_interp_p_3d

def fields_step(self, dt):

    if(self.performance_test_flag == True):
        tic = af.time()
    
    if(self.physical_system.params.field_on == True):
        self.fields_solver.evolve_fields(dt)

    f_interp_p_3d(self, dt)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldstep += toc - tic
    
    return
