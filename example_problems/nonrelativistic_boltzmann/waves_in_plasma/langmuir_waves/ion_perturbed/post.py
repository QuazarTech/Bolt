import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 20, 10
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'bwr'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
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

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
q1  = domain.q1_start + (0.5 + np.arange(domain.N_q1)) * dq1
dp1 = (domain.p1_end - domain.p1_start) / domain.N_p1
p1  = domain.p1_start + (0.5 + np.arange(domain.N_p1)) * dp1

p1, q1 = np.meshgrid(p1, q1)

dt = params.N_cfl * dq1 \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array  = np.arange(0, params.t_final + dt, dt)

h5f       = h5py.File('dump/0000.h5', 'r')
f_initial = h5f['distribution_function'][:]
h5f.close()

maxf_electron_initial = af.max(f_initial[0, :, :domain.N_p1])
minf_electron_initial = af.min(f_initial[0, :, :domain.N_p1])

maxf_ion_initial = af.max(f_initial[0, :, -domain.N_p1:])
minf_ion_initial = af.min(f_initial[0, :, -domain.N_p1:])

maxf_electron = maxf_electron_initial
minf_electron = minf_electron_initial

maxf_ion = maxf_ion_initial
minf_ion = minf_ion_initial

# Traversal to determine variation:
for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0):

        h5f = h5py.File('dump/%04d'%time_index + '.h5', 'r')
        f   = h5f['distribution_function'][:]
        h5f.close()

        f_electron = f[0, :, :domain.N_p1].reshape(domain.N_q1, domain.N_p1)
        f_ion      = f[0, :, -domain.N_p1:].reshape(domain.N_q1, domain.N_p1)

        if(af.max(f_electron)>maxf_electron):
            maxf_electron = af.max(f_electron)

        if(af.min(f_electron)<minf_electron):
            minf_electron = af.min(f_electron)

        if(af.max(f_ion)>maxf_ion):
            maxf_ion = af.max(f_ion)

        if(af.min(f_ion)<minf_ion):
            minf_ion = af.min(f_ion)

delta_f_electron_max = maxf_electron - maxf_electron_initial
delta_f_electron_min = minf_electron - minf_electron_initial

delta_f_ion_max = maxf_ion - maxf_ion_initial
delta_f_ion_min = minf_ion - minf_ion_initial

for time_index, t0 in enumerate(time_array):
    
    if(time_index%10 == 0):
        h5f = h5py.File('dump/%04d'%time_index + '.h5', 'r')
        f   = h5f['distribution_function'][:]
        h5f.close()

        df_electron = (f - f_initial)[0, :, :domain.N_p1].reshape(domain.N_q1, domain.N_p1)
        df_ion      = (f - f_initial)[0, :, -domain.N_p1:].reshape(domain.N_q1, domain.N_p1)

        fig = pl.figure()

        ax1 = fig.add_subplot(1,2,1)
        # ax1.set_aspect('equal')
        c1 = ax1.contourf(p1, q1, df_electron, np.linspace(delta_f_electron_min, delta_f_electron_max, 100))
        ax1.set_xlabel(r'$v$')
        ax1.set_ylabel(r'$x$')

        ax2 = fig.add_subplot(1,2,2)
        # ax2.set_aspect('equal')
        c2 = ax2.contourf(p1, q1, df_ion, np.linspace(delta_f_ion_min, delta_f_ion_max, 100))
        ax2.set_xlabel(r'$v$')
        ax2.set_ylabel(r'$x$')

        fig.suptitle('Time = %.2f'%(t0 - dt))
        fig.colorbar(c1, ax = ax1)
        fig.colorbar(c2, ax = ax2)
        pl.savefig('images/' + '%04d'%time_index + '.png')
        pl.close(fig)
        pl.clf()

h5f = h5py.File('data.h5', 'r')
E_data_ls    = h5f['E_ls'][:]
E_data_nls   = h5f['E_nls'][:]
rho_data_ls  = h5f['n_ls'][:]
rho_data_nls = h5f['n_nls'][:]
time_array   = h5f['time'][:]
h5f.close()

pl.plot(time_array, rho_data_nls[:, 0], '--', color = 'C3', label = 'Electrons')
pl.plot(time_array, rho_data_nls[:, 1], color = 'C0', label='Ions')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_nonlinear.png')
pl.clf()

pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'C3', label = 'Electrons')
pl.plot(time_array, rho_data_ls[:, 1], color = 'C0', label='Ions')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_linear.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 0], color = 'C3', label = 'Electrons(Nonlinear Solver)')
pl.plot(time_array, rho_data_nls[:, 1], color = 'C0', label = 'Ions(Nonlinear Solver)')
pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'C3', label = 'Electrons(Linear Solver)')
pl.plot(time_array, rho_data_ls[:, 1], '--', color = 'C0', label = 'Ions(Linear Solver)')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 0], label = 'Nonlinear Solver')
pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_e.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 1], label = 'Nonlinear Solver')
pl.plot(time_array, rho_data_ls[:, 1], '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_i.png')
pl.clf()

pl.plot(time_array, E_data_nls, label = 'Nonlinear Solver')
pl.plot(time_array, E_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'SUM($|E|^2$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('E.png')
pl.clf()
