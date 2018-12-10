#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_v import integral_over_v

def density(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f, integral_measure))

def mom_rdot_bulk(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * rdot, integral_measure))

def mom_thetadot_bulk(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * thetadot, integral_measure))

def mom_phidot_bulk(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * phidot, integral_measure))

def energy(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(0.5 * f * (rdot**2 + thetadot**2 + phidot**2),
                           integral_measure
                          )
          )

def energy_q1(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(0.5 * f * rdot**2,
                           integral_measure
                          )
          )

def energy_q2(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(0.5 * f * thetadot**2,
                           integral_measure
                          )
          )

def energy_q3(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(0.5 * f * phidot**2,
                           integral_measure
                          )
          )

def q_q1(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * rdot * (rdot**2 + thetadot**2 + phidot**2),
                           integral_measure
                          )
          )

def q_q2(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * thetadot * (rdot**2 + thetadot**2 + phidot**2),
                           integral_measure
                          )
          )

def q_q3(f, rdot, thetadot, phidot, integral_measure):
    return(integral_over_v(f * phidot * (rdot**2 + thetadot**2 + phidot**2),
                           integral_measure
                          )
          )
