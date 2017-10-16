import arrayfire as af

def minmod(x, y, z):
    min_of_all = af.min(af.min(af.abs(x), af.abs(y)), af.abs(z))

    # af.sign(x) = 1 for x<0 and sign(x) for x>0:
    signx = 1 - 2 * af.sign(x)
    signy = 1 - 2 * af.sign(y)
    signz = 1 - 2 * af.sign(z)
    
    result = 0.25 * af.abs(signx + signy ) * (signx + signz ) * min_of_all

    af.eval(result)
    return result

def slopeMM(axis, dx, input):
    filter1D = {1,-1, 0, 0, 1,-1};
  
    if(axis == 0):
        filter = af.Array([3, 1, 1, 2, filter1D])/dx;

    elif(axis == 1):
        filter = af.Array([1, 3, 1, 2, filter1D])/dx;

    elif(axis == 2):
        filter = af.Array([1, 1, 3, 2, filter1D])/dx;
 
    dvar_dX = af.convolve(input, filter);

    forward_diff  = dvar_dX[:, :, :, 0]
    backward_diff = dvar_dX[:, :, :, 1]
    central_diff  = backward_diff + forward_diff

    slopeLimTheta = params.slopeLimTheta

    left   = slopeLimTheta * backward_diff
    center = 0.5 * central_diff
    right  = slopeLimTheta * forward_diff

    return(minmod(left, center, right))
