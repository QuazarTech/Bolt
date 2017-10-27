import arrayfire as af

# Adapted from grim(by Chandra et al.):
def get_LR_states_ppm(input_array, dim):

    if(dim == 'q1'):

        x0_shift = 2;  y0_shift = 0
        x1_shift = 1;  y1_shift = 0
        x3_shift = -1; y3_shift = 0
        x4_shift = -2; y4_shift = 0

    elif(dim == 'q2'):
        x0_shift = 0; y0_shift = 2
        x1_shift = 0; y1_shift = 1
        x3_shift = 0; y3_shift = -1
        x4_shift = 0; y4_shift = -2
  
    y0 = af.shift(input_array, x0_shift, y0_shift)
    y1 = af.shift(input_array, x1_shift, y1_shift)
    y2 = input_array;
    y3 = af.shift(input_array, x3_shift, y3_shift)
    y4 = af.shift(input_array, x4_shift, y4_shift)
    
    # Approximants for slopes
    d0 = 2 * (y1-y0)
    d1 = 2 * (y2-y1)
    d2 = 2 * (y3-y2)
    d3 = 2 * (y4-y3)

    D1 = 0.5 * (y2-y0)
    D2 = 0.5 * (y3-y1)
    D3 = 0.5 * (y4-y2)
  
    cond_zero_slope1 = (d1 * d0 <= 0)
    sign1            = (D1 > 0) * 2 - 1
    DQ1              = (1 - cond_zero_slope1) * sign1 * \
                       af.minof(af.abs(D1),af.minof(af.abs(d0),af.abs(d1)))

    cond_zero_slope2 = (d2 * d1 <= 0)
    sign2            = (D2 > 0) * 2 - 1
    DQ2              = (1 - cond_zero_slope2) * sign2 * \
                       af.minof(af.abs(D2),af.minof(af.abs(d1),af.abs(d2)))

    cond_zero_slope3 = (d3 * d2 <= 0)
    sign3            = (D3 > 0) * 2 - 1;
    DQ3              = (1 - cond_zero_slope3) * sign3 * \
                       af.minof(af.abs(D3),af.minof(af.abs(d2),af.abs(d3)))
  
    # Base high-order PPM reconstruction
    left_value  = 0.5 * (y2 + y1) - (DQ2-DQ1)/6
    right_value = 0.5 * (y3 + y2) - (DQ3-DQ2)/6
  
    # Corrections
    corr1 = ((right_value - y2) * (y2 - left_value) <= 0)
    qd    = right_value - left_value
    qe    = 6 * (y2 - 0.5 * (right_value + left_value))
    corr2 = (qd * (qd - qe) < 0)
    corr3 = (qd * (qd + qe) < 0)
    
    left_value  = left_value  * (1 - corr1) + corr1 * y2;
    right_value = right_value * (1 - corr1) + corr1 * y2;
  
    left_value  = left_value  * (1 - corr2) + corr2 * (3 * y2 - 2 * right_value)
    right_value = right_value * corr2 + (1 - corr2) * right_value * (1 - corr3) + \
                  (1 - corr2) * corr3 * (3 * y2 - 2 * left_value)

    return(left_value, right_value)

def reconstruct_ppm(f, C_q1, C_q2):

    multiply = lambda a, b: a * b

    left_plus_eps_flux, right_minus_eps_flux = get_LR_states_ppm(af.broadcast(multiply, f, C_q1), 'q1')
    bot_plus_eps_flux,  top_minus_eps_flux   = get_LR_states_ppm(af.broadcast(multiply, f, C_q2), 'q2')

    return(left_plus_eps_flux, right_minus_eps_flux,
           bot_plus_eps_flux, top_minus_eps_flux
          )
