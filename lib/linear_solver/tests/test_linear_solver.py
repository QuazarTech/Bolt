#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np 

from lib.physical_system import physical_system
from lib.linear_solver.linear_solver import linear_solver

import test_files.domain as domain
import test_files.periodic_bcs as periodic_bcs
import test_files.dirichlet_bcs as dirichlet_bcs

import test_files.params as params
import test_files.initialize as initialize
import test_files.source as source
import test_files.moment_defs as moment_defs
import test_files.advection_terms as advection_terms

class test():
  def __init__(self):
    self.q1_start = np.random.rand(0, 5)


def test