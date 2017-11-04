#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af


def RK2_step(self, dt):
    """
    Evolves the source/sink term specified by the user
    df/dt = source_sink_term
    using RK2 time stepping.
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    f_initial = self.f  # Storing the value at the start

    if (self.testing_source_flag == False):
        args = (self.q1_center, self.q2_center, self.p1, self.p2,
                self.p3, self.compute_moments, self.physical_system.params
               )

    else:
        args = ()

    # Obtaining value at midpoint(dt/2)
    self.f = self.f + self._source(self.f, *args) * (dt / 2)
    self.f = f_initial + self._source(self.f, *args) * dt

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic
        
    return


def RK4_step(self, dt):
    """
    Evolves the source/sink term specified by the user
    df/dt = source_sink_term
    using RK4 time stepping.
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    f_initial = self.f  # Storing the value at the start

    if (self.testing_source_flag == False):
        args = (self.q1_center, self.q2_center, self.p1, self.p2,
                self.p3, self.compute_moments, self.physical_system.params
               )

    else:
        args = ()

    k1     = self._source(self.f, *args)
    self.f = f_initial + 0.5 * k1 * dt
    k2     = self._source(self.f, *args)
    self.f = f_initial + 0.5 * k2 * dt
    k3     = self._source(self.f, *args)
    self.f = f_initial + k3 * dt
    k4     = self._source(self.f, *args)

    self.f = f_initial + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    af.eval(self.f)
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic

    return


def RK6_step(self, dt):
    """
    Evolves the source/sink term specified by the user
    df/dt = source_sink_term
    using RK6 time stepping.
    """
    if(self.performance_test_flag == True):
        tic = af.time()
    
    f_initial = self.f  # Storing the value at the start

    if (self.testing_source_flag == False):
        args = (self.q1_center, self.q2_center, self.p1, self.p2,
                self.p3, self.compute_moments, self.physical_system.params
               )

    else:
        args = ()

    k1     = self._source(self.f, *args)
    self.f = f_initial + 0.25 * k1 * dt
    
    k2     = self._source(self.f, *args)
    self.f = f_initial + (3 / 32) * (k1 + 3 * k2) * dt
    
    k3     = self._source(self.f, *args)
    self.f = f_initial + (12 / 2197) * (161 * k1 - 600 * k2 + 608 * k3) * dt
    
    k4     = self._source(self.f, *args)
    self.f = f_initial + (1 / 4104) * (  8341 * k1 - 32832 * k2
                                       + 29440 * k3 - 845 * k4
                                      ) * dt

    k5     = self._source(self.f, *args)
    self.f = f_initial + (- (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3
                          + (1859 / 4104) * k4 - (11 / 40) * k5
                         ) * dt

    k6     = self._source(self.f, *args)
    self.f = f_initial + 1 / 5 * (  (16 / 27) * k1 + (6656 / 2565) * k3
                                  + (28561 / 11286) * k4 - (9 / 10) * k5
                                  + (2 / 11) * k6
                                 ) * dt

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic

    return
