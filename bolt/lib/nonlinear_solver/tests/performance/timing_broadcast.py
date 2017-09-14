"""
This file compares the time when operations are performed
using af.broadcast, and without.
"""

import arrayfire as af
af.set_backend('cpu') 
af.info()

def add(a, b):
    return(a + b)

print('With tiling:')

a1 = af.randu(32, 32)
a2 = af.randu(1, 1, 32**3)

a1_tiled = af.tile(a1, 1, 1, 32**3)
a2_tiled = af.tile(a2, 32, 32)

tic = af.time()

a3 = add(a1_tiled, a2_tiled)
a4 = add(a2_tiled, a3)
a5 = add(a3, a4)
a6 = add(a4, a5)

af.eval(a3, a4, a5, a6)

af.sync()
toc = af.time()

print('Kernel Compilation Time =', toc - tic)

tic = af.time()

for i in range(100):
    a3 = add(a1_tiled, a2_tiled)
    a4 = add(a2_tiled, a3)
    a5 = add(a3, a4)
    a6 = add(a4, a5)

    af.eval(a3, a4, a5, a6)

af.sync()
toc = af.time()

print('Time for 100 iterations:', toc - tic)

print('With af.broadcast:')

tic = af.time()
a3  = af.broadcast(add, a1, a2)
a4  = af.broadcast(add, a2, a3)
a5  = af.broadcast(add, a3, a4)
a6  = af.broadcast(add, a4, a5)

af.eval(a3, a4, a5, a6)

af.sync()
toc = af.time()

print('Kernel Compilation Time =', toc - tic)

tic = af.time()

for i in range(100):

    a3  = af.broadcast(add, a1, a2)
    a4  = af.broadcast(add, a2, a3)
    a5  = af.broadcast(add, a3, a4)
    a6  = af.broadcast(add, a4, a5)
    
    af.eval(a3, a4, a5, a6)

af.sync()
toc = af.time()

print('Time for 100 iterations:', toc - tic)
