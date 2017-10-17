import arrayfire as af

def minmod(x, y, z):
    min_of_all = af.min(af.min(af.abs(x), af.abs(y)), af.abs(z))

    # af.sign(x) = 1 for x<0 and sign(x) for x>0:
    signx = 1 - 2 * af.sign(x)
    signy = 1 - 2 * af.sign(y)
    signz = 1 - 2 * af.sign(z)
    
    result = 0.25 * af.abs(signx + signy) * (signx + signz) * min_of_all

    af.eval(result)
    return result

def slopeMM(self, axis):
  
    if(axis == 0):
        f_i_plus_one  = af.shift(self.f * self.C_q1, -1)
        f_i_minus_one = af.shift(self.f * self.C_q1,  1)

        forward_diff  = (f_i_plus_one - self.f * self.C_q1 )/self.dq1
        backward_diff = (self.f * self.C_q1 - f_i_minus_one)/self.dq1
        central_diff  = backward_diff + forward_diff

    elif(axis == 1):
        f_i_plus_one  = af.shift(self.f * self.C_q2, 0, -1)
        f_i_minus_one = af.shift(self.f * self.C_q2, 0,  1)

        forward_diff  = (f_i_plus_one - self.f * self.C_q2 )/self.dq2
        backward_diff = (self.f * self.C_q2 - f_i_minus_one)/self.dq2
        central_diff  = backward_diff + forward_diff

    slope_lim_theta = params.slope_lim_theta

    left   = slope_lim_theta * backward_diff
    center = 0.5 * central_diff
    right  = slope_lim_theta * forward_diff

    return(minmod(left, center, right))

def reconstructMM(self):

    slope = slopeMM(self, 0)

    left_flux  = self.f * self.C_q1 - 0.5 * slope
    right_flux = self.f * self.C_q1 + 0.5 * slope

    slope = slopeMM(self, 1)

    bot_flux = self.f * self.C_q2 - 0.5 * slope
    top_flux = self.f * self.C_q2 + 0.5 * slope

    return(left_flux, right_flux, bot_flux, top_flux)
