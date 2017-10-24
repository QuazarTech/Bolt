import arrayfire as af

# Adapted from grim(by Chandra et al.):
def minmod(x, y, z):

    min_of_all = af.minof(af.minof(af.abs(x),af.abs(y)), af.abs(z))

    # af.sign(x) = 1 for x<0 and sign(x) for x>0:
    signx = 1 - 2 * af.sign(x)
    signy = 1 - 2 * af.sign(y)
    signz = 1 - 2 * af.sign(z)
    
    result = 0.25 * af.abs(signx + signy) * (signx + signz) * min_of_all

    af.eval(result)
    return result

def slope_minmod(input_array, dim):
  
    if(dim == 'q1'):
        
        f_i_plus_one  = af.shift(input_array, -1)
        f_i_minus_one = af.shift(input_array,  1)

    elif(dim == 'q2'):

        f_i_plus_one  = af.shift(input_array, 0, -1)
        f_i_minus_one = af.shift(input_array, 0,  1)

    forward_diff  = (f_i_plus_one - input_array  )
    backward_diff = (input_array  - f_i_minus_one)
    central_diff  = backward_diff + forward_diff

    slope_lim_theta = 2

    left   = slope_lim_theta * backward_diff
    center = 0.5 * central_diff
    right  = slope_lim_theta * forward_diff

    return(minmod(left, center, right))

def reconstruct_minmod(f, C_q1, C_q2):

    multiply = lambda a, b: a * b
    slope = slope_minmod(af.broadcast(multiply, f, C_q1), 'q1')

    left_plus_eps_flux   = af.broadcast(multiply, f, C_q1) - 0.5 * slope
    right_minus_eps_flux = af.broadcast(multiply, f, C_q1) + 0.5 * slope

    slope = slope_minmod(af.broadcast(multiply, f, C_q2), 'q2')

    bot_plus_eps_flux  = af.broadcast(multiply, f, C_q2) - 0.5 * slope
    top_minus_eps_flux = af.broadcast(multiply, f, C_q2) + 0.5 * slope

    return(left_plus_eps_flux, right_minus_eps_flux,
           bot_plus_eps_flux, top_minus_eps_flux
          )
