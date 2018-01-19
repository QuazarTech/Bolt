************
Installation
************

Downloading the Source
-----------------------

Bolt is distributed using the git version control system, and is hosted on Github. The repository can be cloned using::

    git clone https://github.com/QuazarTech/Bolt.git


Dependencies
-------------

Overview
^^^^^^^^

Bolt has a hard dependency on Python 3+ and the following Python packages:

1. `mpi4py <http://mpi4py.scipy.org/>`_ 
2. `numpy <http://www.numpy.org/>`_ 
3. `h5py <http://www.h5py.org/>`_ 
4. `pytest <https://pypi.python.org/pypi/pytest>`_
5. `scipy <https://www.scipy.org/>`_
6. `matplotlib <https://http://matplotlib.org/>`_
7. `petsc4py <https://bitbucket.org/petsc/petsc4py>`_ 
8. `arrayfire <https://github.com/arrayfire/arrayfire-python>`_ 

Before installing the above python packages, the following libraries need to be installed so that their python wrappers can function: 

Building ArrayFire
^^^^^^^^^^^^^^^^^^

- Clone the `arrayfire <https://github.com/arrayfire/arrayfire>`_ repository
- Build using the instructions that have been provided `here <https://github.com/arrayfire/arrayfire/wiki/Build-Instructions-for-Linux>`_ 

Building PETSc
^^^^^^^^^^^^^^

- Clone the `petsc <https://bitbucket.org/petsc/petsc>`_ repository
- We suggest that you install PETSc using the following::

    ./configure --prefix=/path/to/petsc_installation/ --with-debugging=0 COPTFLAGS="-O3 -march=native" CXXOPTFLAG S="-O3 -march=native" --with-hdf5=1 --download-hdf5 --with-clean=1 --with-memalign=64 --known-level1-dcache-size=32768 --known-level1-dcache-linesize=64 --known-level1-dcache-assoc=8 --with-hypre=1 --download-mpich=1 --with-64-bit-indices

- If you are keen on modifying the above build parameters, detailed instructions for the same may be found `here <http://www.mcs.anl.gov/petsc/documentation/installation.html>`_

Below are instructions for building the PETSc stack on a few machines that we've tested on:

- On BRC HPC Savio::

    python2 './configure' '--with-debugging=0' 'COPTFLAGS=-O3 -qopt-report=5 -qopt-report-phase=vec -xhost' 'CXXOPTFLAGS=-O3 -qopt-report=5 -qopt-report-phase=vec -xhost' '--with-hdf5=1' '--with-clean=1' '--with-mpi-dir=/global/software/sl-6.x86_64/modules/intel/2016.1.150/openmpi/1.10.2-intel/' '--with-blas-lapack-dir=/global/software/sl-6.x86_64/modules/langs/intel/2016.1.150/mkl/lib/intel64' '--with-memalign=64' '--known-level1-dcache-size=32768' --known-level1-dcache-linesize=64' 'known-level1-dcache-assoc=8' '--with-hypre=1' '--download-hypre=1' '--with-64-bit-indices'


Installation
-------------

Before running Bolt it is first necessary to either install the software using the provided ``setup.py`` installer(TODO) or add the root directory to ``PYTHONPATH`` using::

    user@computer ~/Bolt$ export PYTHONPATH=.:$PYTHONPATH

Once the build of ArrayFire and PETSc is completed install the python dependencies using::

    user@computer ~/Bolt$ pip install -r requirements.txt
