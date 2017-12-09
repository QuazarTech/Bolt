import arrayfire as af

# Adapted from grim(by Chandra et al.):
def reconstruct_weno5(input_array, axis):
    
    eps = 1e-17;

    if(axis == 0):

        x0_shift = 2;  y0_shift = 0; z0_shift = 0
        x1_shift = 1;  y1_shift = 0; z1_shift = 0
        x3_shift = -1; y3_shift = 0; z3_shift = 0
        x4_shift = -2; y4_shift = 0; z4_shift = 0

    elif(axis == 1):

        x0_shift = 0; y0_shift = 2;  z0_shift = 0
        x1_shift = 0; y1_shift = 1;  z1_shift = 0
        x3_shift = 0; y3_shift = -1; z3_shift = 0
        x4_shift = 0; y4_shift = -2; z4_shift = 0
  
    elif(axis == 2):

        x0_shift = 0; y0_shift = 0; z0_shift = 2
        x1_shift = 0; y1_shift = 0; z1_shift = 1
        x3_shift = 0; y3_shift = 0; z3_shift = -1
        x4_shift = 0; y4_shift = 0; z4_shift = -2

    else:
        raise Exception('Invalid choice for axis')

    y0 = af.shift(input_array, x0_shift, y0_shift, z0_shift)
    y1 = af.shift(input_array, x1_shift, y1_shift, z1_shift)
    y2 = input_array;
    y3 = af.shift(input_array, x3_shift, y3_shift, z3_shift)
    y4 = af.shift(input_array, x4_shift, y4_shift, z4_shift)

    # Compute smoothness operators
    beta1 = (( 4/3) * y0 * y0 - (19/3) * y0 * y1 +
             (25/3) * y1 * y1 + (11/3) * y0 * y2 -
             (31/3) * y1 * y2 + (10/3) * y2 * y2
            ) + eps * (1.0 + af.abs(y0) + af.abs(y1) + af.abs(y2))
  
    beta2 = (( 4/3) * y1 * y1 - (19/3) * y1 * y2 +
             (25/3) * y2 * y2 + (11/3) * y1 * y3 -
             (31/3) * y2 * y3 + (10/3) * y3 * y3
            ) + eps * (1.0 + af.abs(y1) + af.abs(y2) + af.abs(y3))
  
    beta3 = (( 4/3) * y2 * y2 - (19/3) * y2 * y3 +
             (25/3) * y3 * y3 + (11/3) * y2 * y4 -
             (31/3) * y3 * y4 + (10/3) * y4 * y4
            ) + eps * (1.0 + af.abs(y2) + af.abs(y3) + af.abs(y4))
  
    # Compute weights
    w1r = 1 / (16 * beta1 * beta1);
    w2r = 5 / ( 8 * beta2 * beta2);
    w3r = 5 / (16 * beta3 * beta3);
    
    w1l = 5 / (16 * beta1 * beta1);
    w2l = 5 / ( 8 * beta2 * beta2);
    w3l = 1 / (16 * beta3 * beta3);
    
    denl = w1l + w2l + w3l;
    denr = w1r + w2r + w3r;

    # Substencil Interpolations
    u1r =  0.375 * y0 - 1.25 * y1 + 1.875 * y2;
    u2r = -0.125 * y1 + 0.75 * y2 + 0.375 * y3;
    u3r =  0.375 * y2 + 0.75 * y3 - 0.125 * y4;
    
    u1l = -0.125 * y0 + 0.75 * y1 + 0.375 * y2;
    u2l =  0.375 * y1 + 0.75 * y2 - 0.125 * y3;
    u3l =  1.875 * y2 - 1.25 * y3 + 0.375 * y4;

    # Reconstruction:
    left_value  = (w1l * u1l + w2l * u2l + w3l * u3l) / denl;
    right_value = (w1r * u1r + w2r * u2r + w3r * u3r) / denr;

    af.eval(left_value, right_value)
    return(left_value, right_value)
