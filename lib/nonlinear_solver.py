#!/usr/bin/env python 

import arrayfire as af 
import numpy as np 

from mpi4py import MPI
from petsc4py import PETSc 

class nonlinear_solver(object):