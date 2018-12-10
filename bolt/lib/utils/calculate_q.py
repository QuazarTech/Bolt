#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def calculate_q_center(q1_start, q2_start, 
                       N_q1, N_q2, N_g,
                       dq1, dq2
                      ):
    """
    Initializes the cannonical variables q1, q2 using a centered
    formulation. 

    Returns in q_expanded form.
    """
    i_q1 = (0.5 + np.arange(-N_g, N_q1 + N_g))
    i_q2 = (0.5 + np.arange(-N_g, N_q2 + N_g))

    q1_center = q1_start + i_q1 * dq1
    q2_center = q2_start + i_q2 * dq2

    q2_center, q1_center = np.meshgrid(q2_center, q1_center)
    q1_center, q2_center = af.to_array(q1_center), af.to_array(q2_center)

    # To bring the data structure to the default form:(N_p, N_s, N_q1, N_q2)
    q1_center = af.reorder(q1_center, 3, 2, 0, 1)
    q2_center = af.reorder(q2_center, 3, 2, 0, 1)

    af.eval(q1_center, q2_center)
    return (q1_center, q2_center)

def calculate_q_left_center(q1_start, q2_start, 
                            N_q1, N_q2, N_g,
                            dq1, dq2
                           ):

    i_q1 = np.arange(-N_g, N_q1 + N_g)
    i_q2 = (0.5 + np.arange(-N_g, N_q2 + N_g))

    q1_left_center = q1_start + i_q1 * dq1
    q2_left_center = q2_start + i_q2 * dq2

    q2_left_center, q1_left_center = np.meshgrid(q2_left_center, q1_left_center)
    q2_left_center, q1_left_center = af.to_array(q2_left_center), af.to_array(q1_left_center)

    # To bring the data structure to the default form:(N_p, N_s, N_q1, N_q2)
    q1_left_center = af.reorder(q1_left_center, 3, 2, 0, 1)
    q2_left_center = af.reorder(q2_left_center, 3, 2, 0, 1)

    af.eval(q1_left_center, q2_left_center)
    return (q1_left_center, q2_left_center)

def calculate_q_center_bot(q1_start, q2_start, 
                           N_q1, N_q2, N_g,
                           dq1, dq2
                          ):

    i_q1 = (0.5 + np.arange(-N_g, N_q1 + N_g))
    i_q2 = np.arange(-N_g, N_q2 + N_g)

    q1_center_bot = q1_start + i_q1 * dq1
    q2_center_bot = q2_start + i_q2 * dq2

    q2_center_bot, q1_center_bot = np.meshgrid(q2_center_bot, q1_center_bot)
    q2_center_bot, q1_center_bot = af.to_array(q2_center_bot), af.to_array(q1_center_bot)

    # To bring the data structure to the default form:(N_p, N_s, N_q1, N_q2)
    q1_center_bot = af.reorder(q1_center_bot, 3, 2, 0, 1)
    q2_center_bot = af.reorder(q2_center_bot, 3, 2, 0, 1)

    af.eval(q1_center_bot, q2_center_bot)
    return (q1_center_bot, q2_center_bot)

def calculate_q_left_bot(q1_start, q2_start, 
                         N_q1, N_q2, N_g,
                         dq1, dq2
                        ):

    i_q1 = np.arange(-N_g, N_q1 + N_g)
    i_q2 = np.arange(-N_g, N_q2 + N_g)

    q1_left_bot = q1_start + i_q1 * dq1
    q2_left_bot = q2_start + i_q2 * dq2

    q2_left_bot, q1_left_bot = np.meshgrid(q2_left_bot, q1_left_bot)
    q2_left_bot, q1_left_bot = af.to_array(q2_left_bot), af.to_array(q1_left_bot)

    # To bring the data structure to the default form:(N_p, N_s, N_q1, N_q2)
    q1_left_bot = af.reorder(q1_left_bot, 3, 2, 0, 1)
    q2_left_bot = af.reorder(q2_left_bot, 3, 2, 0, 1)

    af.eval(q1_left_bot, q2_left_bot)
    return (q1_left_bot, q2_left_bot)
