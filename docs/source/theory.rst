******
Theory
******

In :math:`\texttt{Bolt}`, the defined system is evolved by stepping the complete probability distribution function, from which physical parameters of interest about the system may be obtained by coarse-graining the system via use of the ``compute_moments`` method. The implementation allows us to deal with a wide range of boundary conditions with ease, in addition to capturing the dynamics of short-range interactions via a collision operator(input as a source). :math:`\texttt{Bolt}` is capable of performing accurate simulations of systems that are governed by the form:

.. math::
  \frac{\partial f}{\partial t} + A_{q1} \frac{\partial f}{\partial q_1} + A_{q2} \frac{\partial f}{\partial q_2} + A_{p1} \frac{\partial f}{\partial p_1} + A_{p2} \frac{\partial f}{\partial p_2} + A_{p3} \frac{\partial f}{\partial p_3} = S(f)

``Bolt`` can make use of the finite-volume method and/or the non-conservative semi-lagrangian method. 

Finite Volume Method
====================

To explore this method in detail, we'll first need to define the generalized conservative equations:

.. math::
    \frac{\partial f}{\partial t} + \frac{\partial (C_{q1} f)}{\partial q_1} + \frac{\partial (C_{q2} f)}{\partial q_2} +  \frac{\partial (C_{p1} f)}{\partial p_1} + \frac{\partial (C_{p2} f)}{\partial p_2} + \frac{\partial (C_{p3} f)}{\partial p_3} = S(f)

The conservative equations are multiplied by the volume element of a discrete grid zone in phase space :math:`\Delta v = dq_1 dq_2 dp_1 dp_2 dp_3`, and using the divergence theorem gives use the finite volume formulation:

.. math::
    \partial_t \bar{f} + \frac{{\bar{F}_{q1}}^{q-right} - {\bar{F}_{q1}}^{q-left}}{\Delta dq1} +  \frac{{\bar{F}_{q2}}^{q-top} - {\bar{F}_{q2}}^{q-bottom}}{\Delta dq2} + \frac{{\bar{F}_{p1}}^{p-right} - {\bar{F}_{p1}}^{p-left}}{\Delta dp1} \\ + \frac{{\bar{F}_{p2}}^{p-top} - {\bar{F}_{p2}}^{p-bottom}}{\Delta dp2} + \frac{{\bar{F}_{p3}}^{p-front} - {\bar{F}_{p3}}^{p-back}}{\Delta dp3} = \bar{S}

Where :math:`\bar{f} = (\int f \Delta v)/\int \Delta v`, :math:`\bar{S} = (\int S \Delta v)/\int \Delta v`, :math:`\bar{F}_{q1} = (\int f C_{q1} dq_2 dp_1 dp_2 dp_3)/\int dq_2 dp_1 dp_2 dp_3`, :math:`\bar{F}_{q2} = (\int f C_{q2} dq_1 dp_1 dp_2 dp_3)/\int dq_2 dp_1 dp_2 dp_3`, :math:`\bar{F}_{p1} = (\int f C_{p1} dq_1 dq_2 dp_2 dp_3)/\int dq_1 dq_2 dp_2 dp_3`, :math:`\bar{F}_{p2} = (\int f C_{p2} dq_1 dq_2 dp_1 dp_3)/\int dq_1 dq_2 dp_1 dp_3`, :math:`\bar{F}_{p3} = (\int f C_{p3} dq_1 dq_2 dp_1 dp_2)/\int dq_1 dq_2 dp_1 dp_2`.

The locations right, left, top, bottom, front, and back are mentioned whether they are considered in q-space or p-space since we are considering 5-dimensional phase space here. The appropriate value at the boundaries is obtained by using a reconstruction operator which takes in the cell-centered values and constructs a polynomial interpolant from which the edge states can be computed. The time derivative term :math:`\partial_t \bar{f}` is then passed to an appropriate integrator to evolve the system in time. 

Semi-Lagrangian Method
======================

In this approach, a probability distribution function is given a grid-based Eulerian representation which is then evolved via Lagrangian dynamics. The CFL time step restriction of a regular finite difference or finite volume scheme is removed in a semi-Lagrangian framework, allowing for a cheaper and more flexible numerical realization. However, the downside is that the method is non-conservative

A detailed overview of the advective semi-Lagrangian method is given in:

- `The integration of the Vlasov equation in configuration space`. Cheng, C. Z., & Knorr, G. (1976). Journal of Computational Physics, 22(3), 330-351.<http://www.sciencedirect.com/science/article/pii/002199917690053X>
