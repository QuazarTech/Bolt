import arrayfire as af

def af_to_petsc_glob_array(domain_metadata, af_array, glob_array):

    # domaint_metadata obj should contain the following
    i_q1_start = domain_metadata.i_q1_start # start index of bulk domain
    i_q1_end   = domain_metadata.i_q1_end   # end index, before ghost zones

    i_q2_start = domain_metadata.i_q2_start
    i_q2_end   = domain_metadata.i_q2_end

    tmp_array = af.flat(af_array[:, :, 
                                 i_q1_start:i_q1_end,
                                 i_q2_start:i_q2_end
                                ]
                       )

    tmp_array.to_ndarray(glob_array)

    return

def petsc_local_array_to_af(domain_metadata,
                            N_vars_axis_0,
                            N_vars_axis_1,
                            local_array
                           ):

    flat_af_array = af.to_array(local_array)

    #resize to the appropriate shape
    af_array      = af.moddims(flat_af_array,
                               N_vars_axis_0,
                               N_vars_axis_1,
                               domain_metadata.N_q1_local_with_Ng,
                               domain_metadata.N_q2_local_with_Ng
                              )

    return(af_array)
