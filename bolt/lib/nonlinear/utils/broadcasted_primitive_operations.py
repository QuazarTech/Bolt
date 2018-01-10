"""
Here we define fundamental operations such as multiplications
and additions wrapped with the arrayfire broadcast wrapper. 
This is then called in the individual module files when operations
on arrays of different sizes need to be performed.
"""

import arrayfire as af 

@af.broadcast
def add(*args):

    result = 0
    for x in args:
        result += x

    return(result)

@af.broadcast
def multiply(*args):

    result = 1
    for x in args:
        result *= x

    return(result)
