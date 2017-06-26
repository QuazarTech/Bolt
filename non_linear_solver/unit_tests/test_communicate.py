import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import arrayfire as af 
import cks.communicate

class args:
  def __init__():
    pass

class config:
  def __init__():
    pass

config.bc_in_x = 'periodic'
config.bc_in_y = 'periodic'

N_y = np.random.randint(low = 16, high = 512)
N_x = np.random.randint(low = 16, high = 512)

config.N_ghost = np.random.randint(low = 1, high = 5)
args.config    = config

da = PETSc.DMDA().create([N_y, N_x],\
                         dof = 2,\
                         stencil_width = args.config.N_ghost,\
                         boundary_type = (args.config.bc_in_x, args.config.bc_in_y),\
                         stencil_type = 1, \
                        )

glob  = da.createGlobalVector()
local = da.createLocalVector()

local_val = da.getVecArray(local) 

i = 0.5 - config.N_ghost + np.arange(N_x + 2*config.N_ghost)
j = 0.5 - config.N_ghost + np.arange(N_y + 2*config.N_ghost)
x = i/N_x # Taking dx = 1/N_x
y = j/N_y # Taking dy = 1/N_y

print(x)
print(y)

x, y = np.meshgrid(x, y)

x, y = af.to_array(x), af.to_array(y)

check   = af.sin(2*np.pi*x + 4*np.pi*y) # Solution to check against
changed = check.copy()

# Messing the values at the boundary
# changed[-config.N_ghost:0, -config.N_ghost:0] = changed[-config.N_ghost:0, -config.N_ghost:0] + np.random.randint(5000)
# changed[:config.N_ghost, :config.N_ghost]     = changed[:config.N_ghost, :config.N_ghost]     + np.random.randint(5000)

args.f = af.constant(0, N_y + 2*config.N_ghost, N_x + 2*config.N_ghost, 2)

args.f[:, :, 0] = changed
args.f[:, :, 1] = changed

def test_communicate_distribution_function():
  args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
  assert(af.sum(af.abs(args.f[:, :, 0] - check))==0)
