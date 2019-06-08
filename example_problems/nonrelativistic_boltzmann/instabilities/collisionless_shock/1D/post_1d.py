import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl
from petsc4py import PETSc

import domain
import params

from post import return_moment_to_be_plotted
from post import return_field_to_be_plotted
from post import determine_min_max, q1, q2
from post import da_moments, da_fields, moments_vec, fields_vec

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 18, 8
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

t_final = 284 * params.t0
time_array = np.arange(0, t_final + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

print("Determining limits...")
n_min, n_max   = determine_min_max('density', time_array)
v1_min, v1_max = determine_min_max('v1', time_array)
T_min, T_max   = determine_min_max('temperature', time_array)
B1_min, B1_max = determine_min_max('B1', time_array)

for time_index, t0 in enumerate(time_array):

    print("file = ", time_index)
    
    moments_file = 'dump_moments/t=%.3f'%(t0) + '.bin'
    fields_file  = 'dump_fields/t=%.3f'%(t0) + '.bin'

    # Load moments
    viewer = PETSc.Viewer().createBinary(moments_file, 
                                         PETSc.Viewer.Mode.READ, 
                                        )

    moments_vec.load(viewer)
    moments = da_moments.getVecArray(moments_vec) # [N_q1, N_q2, N_moments]

    # Load fields
    viewer = PETSc.Viewer().createBinary(fields_file, 
                                         PETSc.Viewer.Mode.READ, 
                                        )

    fields_vec.load(viewer)
    fields = da_fields.getVecArray(fields_vec) # [N_q1, N_q2, N_fields]

    n  = return_moment_to_be_plotted('density', moments)
    v1 = return_moment_to_be_plotted('v1', moments)
    T  = return_moment_to_be_plotted('temperature', moments)
    B1 = return_field_to_be_plotted('B1', fields)

    fig = pl.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(q2[0, :], n[0, :, 0], color = 'C0', label = 'Electrons')
    ax1.plot(q2[0, :], n[0, :, 1], '--', color = 'C3', label = 'Ions')
    ax1.legend()
    ax1.set_xlabel(r'$y(l_s)$')
    ax1.set_ylabel(r'$n(n_0)$')
    ax1.set_ylim([0.95 * n_min, 1.05 * n_max])

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(q2[0, :], v1[0, :, 0] / params.v0, color = 'C0')
    ax2.plot(q2[0, :], v1[0, :, 1] / params.v0, '--', color = 'C3')
    ax2.set_xlabel(r'$y(l_s)$')
    ax2.set_ylabel(r'$v_x(v_0)$')
    ax2.set_ylim([1.05 * v1_min / params.v0, 1.05 * v1_max / params.v0])

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(q2[0, :], T[0, :, 0] / params.T0, color = 'C0')
    ax3.plot(q2[0, :], T[0, :, 1] / params.T0, '--', color = 'C3')
    ax3.set_xlabel(r'$y(l_s)$')
    ax3.set_ylabel(r'$T(T_0)$')
    ax3.set_ylim([0.95 * T_min / params.T0, 1.05 * T_max / params.T0])

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(q2[0, :], B1[0, :] / params.B0)
    ax4.set_xlabel(r'$y(l_s)$')
    ax4.set_ylabel(r'$B_x(\sqrt{n_0 m_0} v_0)$')
    ax4.set_ylim([0.95 * B1_min / params.B0, 1.05 * B1_max / params.B0])

    # fig.tight_layout()
    fig.suptitle('Time = %.2f'%(t0 / params.t0)+r'$\omega_c^{-1}$=%.2f'%(t0 * params.plasma_frequency)+r'$\omega_p^{-1}$')
    pl.savefig('images/%04d'%time_index + '.png')
    pl.close(fig)
    pl.clf()
