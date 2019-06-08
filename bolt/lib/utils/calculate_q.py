#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def calculate_q(q1_start, q2_start, 
                N_q1, N_q2,
                N_g1, N_g2,
                dq1, dq2
               ):

    i_q1 = np.arange(-N_g1, N_q1 + N_g1)
    i_q2 = np.arange(-N_g2, N_q2 + N_g2)

    q1_left_bot = q1_start + i_q1 * dq1
    q2_left_bot = q2_start + i_q2 * dq2

    q2_left_bot, q1_left_bot = np.meshgrid(q2_left_bot, q1_left_bot)
    q2_left_bot, q1_left_bot = af.to_array(q2_left_bot), af.to_array(q1_left_bot)

    # To bring the data structure to the default form:(N_p, N_s, N_q1, N_q2)
    q1_left_bot = af.reorder(q1_left_bot, 3, 2, 0, 1)
    q2_left_bot = af.reorder(q2_left_bot, 3, 2, 0, 1)

    q1_center_bot  = q1_left_bot + 0.5*dq1
    q2_center_bot  = q2_left_bot

    q1_left_center = q1_left_bot
    q2_left_center = q2_left_bot + 0.5*dq2

    q1_center      = q1_left_bot + 0.5*dq1
    q2_center      = q2_left_bot + 0.5*dq2

    ans = [ [q1_left_bot,    q2_left_bot  ],
            [q1_center_bot,  q2_center_bot], 
            [q1_left_center, q2_center_bot], 
            [q1_center,      q2_center    ]
          ]

    return(ans)
