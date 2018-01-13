"""
Evaluates the Bandwidth of the device run on
using the STREAMS benchmark.
"""

import arrayfire as af

def memory_bandwidth(size, n_reads, n_writes, n_evals, time_elapsed):
    return(size * 8 * (n_reads + n_writes) 
                * n_evals/(time_elapsed*2**30)
          )

def bandwidth_test(n_evals):

    a = af.randu(32, 32, 32**3, dtype = af.Dtype.f64)
    b = af.randu(32, 32, 32**3, dtype = af.Dtype.f64)
    c = a + b
    af.eval(c)
    af.sync()

    tic = af.time()
    for i in range(n_evals):
        c = a + b
        af.eval(c)
    af.sync()
    toc = af.time()

    bandwidth_available = memory_bandwidth(a.elements(), 2, 1,
                                           n_evals, toc - tic
                                          )

    return(bandwidth_available)
