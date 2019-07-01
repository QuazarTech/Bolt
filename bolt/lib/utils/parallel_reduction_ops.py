import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

def global_mean(af_array):
    local_mean  = np.mean(np.array(af_array))
    comm        = PETSc.COMM_WORLD.tompi4py()

    global_mean =  comm.allreduce(sendobj=local_mean, op=MPI.SUM) \
                 / comm.size

    return global_mean

def global_min(af_array):
    local_min  = np.min(np.array(af_array))
    comm       = PETSc.COMM_WORLD.tompi4py()

    global_min = comm.allreduce(sendobj=local_min, op=MPI.MIN)

    return global_min

def global_max(af_array):
    local_max  = np.min(np.array(af_array))
    comm       = PETSc.COMM_WORLD.tompi4py()

    global_max = comm.allreduce(sendobj=local_min, op=MPI.MAX)

    return global_max
