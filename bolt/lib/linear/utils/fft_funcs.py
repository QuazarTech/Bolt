"""
We want the fft operations to be performed along the axis containing
the variation along q1, and q2. Since ArrayFire doesn't have the 
capability to specify axes along which the operations are to be 
performed, here we define our fft2 and ifft2 functions which perform
the operation along the axis 2, 3 of the array using reorders.

NOTE: This is just a temporary workaround. A request has been made to arrayfire
      for the implementation of fft which allows targeting specific axes.
"""

import arrayfire as af

def fft2(array):
    # Reorder from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s) 
    array = af.reorder(array, 2, 3, 0, 1)
    array = af.fft2(array)
    # Reorder back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2) 
    array = af.reorder(array, 2, 3, 0, 1)

    af.eval(array)
    return(array)

def ifft2(array):
    # Reorder from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s) 
    array = af.reorder(array, 2, 3, 0, 1)
    array = af.ifft2(array)
    # Reorder back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2) 
    array = af.reorder(array, 2, 3, 0, 1)

    af.eval(array)
    return(array)
