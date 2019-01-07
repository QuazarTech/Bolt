# Post-Processing

The purpose of this README is to illustrate how the post-processing is carried out. `post.py` contains the routines that one would need for post processing. We illustrate the usage of these routines by going through `post_1d.py`(NOTE: This script needs to be executed in serial. i.e not parallel). Using this file we intend to make a movie showing the variation of the density, bulk velocity, temperature and Bx in x with time:

First declare the time array over which the plotting needs to be carried out:

```
time_array = np.arange(0, params.t_final + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )
```

We need to determine the maximum and minimum values that our quantities of interest take over this time. This is needed to ensure that the y-axis limits remain the same throughout.

```
n_min, n_max   = determine_min_max('density', time_array)
v1_min, v1_max = determine_min_max('v1', time_array)
T_min, T_max   = determine_min_max('temperature', time_array)
B1_min, B1_max = determine_min_max('B1', time_array)
```

Now we move on to the main time loop in which the plotting has been carried out. First the arrays need to be read from the files written by `dump_moments` and `dump_EM_fields`. The files are written in the format `(q2, q1, N_s)`. However, all the functions we use to post-process will expect the input in the form `(q1, q2, N_s)`. For this reason, we swap the 0th and 1st axes of the arrays:

```
h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
moments = np.swapaxes(h5f['moments'][:], 0, 1)
h5f.close()

h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
h5f.close()
```

To get the quantity that needs to be plotted, either `return_moment_to_be_plotted` or `return_field_to_be_plotted` needs to be used. Here, we want to plot density, bulk velocity, temperature and B_x. For the field arrays(i.e E1, E2, E3, B1, B2, B3), `return_field_to_be_plotted` needs to be used. For everything else, we use `return_moment_to_be_plotted`

```
n  = return_moment_to_be_plotted('density', moments)
v1 = return_moment_to_be_plotted('v1', moments)
T  = return_moment_to_be_plotted('temperature', moments)
B1 = return_field_to_be_plotted('B1', fields)
```

Once these arrays have been obtained, the plotting can be carried out depending upon your choice.
