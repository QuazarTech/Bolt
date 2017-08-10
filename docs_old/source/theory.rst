******
Theory
******

Semi-Lagrangian Method
======================

Overview
--------

In this approach, a probability distribution function is given a grid-based 
Eulerian representation which is then evolved via Lagrangian dynamics. The
scheme allows us to deal with a wide range of boundary conditions with ease, 
in addition to capturing the dynamics of short-range interactions via a collision
operator. The CFL time step restriction of a regular finite difference or 
finite volume scheme is removed in a semi-Lagrangian framework, allowing for a cheaper
and more flexible numerical realization.Additionally, this approach exhibits a 
significant degree of element locality, and is thus able to run efficiently on 
modern streaming architectures, such as Graphical Processing Units (GPUs). The
aforementioned properties of this method mean it offers a promising route to
performing affordable, and accurate simulations of systems that are governed by the form:

.. math::

  \frac{\partial f}{\partial t} + A_{q1} \frac{\partial f}{\partial q_1} + A_{q2} \frac{\partial f}{\partial q_2} + A_{p1} \frac{\partial f}{\partial p_1} + A_{p2} \frac{\partial f}{\partial p_2} + A_{p3} \frac{\partial f}{\partial p_3} = g(f)

A detailed overview of the semi-Lagrangian method is given in:

- `The integration of the Vlasov equation in configuration space`. Cheng, C. Z.,
   & Knorr, G. (1976). Journal of Computational Physics, 22(3), 330-351.
   <http://www.sciencedirect.com/science/article/pii/002199917690053X>