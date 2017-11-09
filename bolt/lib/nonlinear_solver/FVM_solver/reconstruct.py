from .reconstruction_methods.minmod import reconstruct_minmod
from .reconstruction_methods.ppm import reconstruct_ppm
from .reconstruction_methods.weno5 import reconstruct_weno5

def reconstruct(self, input_array, dim):

    if(self.physical_system.params.reconstruction_method == 'piecewise-constant'):
        left_face_value  = input_array
        right_face_value = input_array

    elif(self.physical_system.params.reconstruction_method == 'minmod'):
        left_face_value, right_face_value = reconstruct_minmod(input_array, dim)
        
    elif(self.physical_system.params.reconstruction_method == 'ppm'):
        left_face_value, right_face_value = reconstruct_ppm(input_array, dim)

    elif(self.physical_system.params.reconstruction_method == 'weno5'):
        left_face_value, right_face_value = reconstruct_weno5(input_array, dim)

    else:
        raise NotImplementedError('Reconstruction method invalid/not-implemented')

    return(left_face_value, right_face_value)
