# Author: Vitezslav Luzny
# Contact: luzny274@gmail.com

# distutils: language = c++

from enum import Enum

from libcpp cimport bool
from libc.stdint cimport int64_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint16_t
from libcpp cimport float

import ctypes as C
from ctypes.util import find_library

import numpy as np
cimport numpy as np


# Import the convolution functions from c++ 
ctypedef double   my_cfl64
ctypedef float    my_cfl32
ctypedef uint32_t my_cui32 
ctypedef uint16_t my_cui16 

cdef extern from "../cpp/my_conv.cpp":
    cdef void cpp_conv_fl64(int number_of_threads, int64_t sample_count, my_cfl64 * samples, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int64_t subpixels, int64_t psf_res, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cfl64 * intensities, my_cfl64 * PSF_subpx)
    cdef void cpp_conv_fl32(int number_of_threads, int64_t sample_count, my_cfl32 * samples, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int64_t subpixels, int64_t psf_res, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cfl32 * intensities, my_cfl32 * PSF_subpx)
    
    cdef void cpp_conv_ui32(int number_of_threads, int64_t sample_count, my_cui32 * samples, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int64_t subpixels, int64_t psf_res, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cui32 * intensities, my_cui32 * PSF_subpx)
    cdef void cpp_conv_ui16(int number_of_threads, int64_t sample_count, my_cui16 * samples, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int64_t subpixels, int64_t psf_res, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cui16 * intensities, my_cui16 * PSF_subpx)


def convolve(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, PSF_subpx, datatype):
    """
    This function takes seven arguments:

    number_of_threads : int
        Maximum number of threads on which to paralalyze the convolution for loop
    camera_fov_px : int
        Dimension of the output
    particle_positions : numpy.ndarray[np.double_t, ndim=3]
        Array of individual particle positions in different frames, dimensions: (step_count, particle_count, 2 coordinates)
    sample_sizes : numpy.ndarray[np.int32_t, ndim=1]
        Number of particles belonging to a specific sample, dimension: (particle_count)
    intensities : numpy.ndarray[datatype, ndim=1]
        Intensities of individual particles, dimension: (particle_count)
    PSF_subpx : numpy.ndarray[datatype, ndim=4]
        Point Spread Function array, dimensions: (subpixels, subpixels, psf_res, psf_res)
    datatype : type
        datatype of intensities, PSF and output. Possible values: numpy.float64, numpy.float32, numpy.uint32, numpy.uint16

    Returns
    -------
    numpy.ndarray[datatype, ndim=4]
        Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px). If convolution could not happen due to invalid input, it is filled with zeros.
    """

    if not isinstance(number_of_threads, int):
        raise TypeError('number_of_threads must be an integer')
    if not isinstance(camera_fov_px, int):
        raise TypeError('camera_fov_px must be an integer')

    if datatype == np.float64:
        return convolve_fl64(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, PSF_subpx)
    elif datatype == np.float32:
        return convolve_fl32(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, PSF_subpx)
    elif datatype == np.uint32:
        return convolve_ui32(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, PSF_subpx)
    elif datatype == np.uint16:
        return convolve_ui16(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, PSF_subpx)
    else:
        raise TypeError('Datatype argument must be either numpy.float64, numpy.float32, numpy.uint32 or numpy.uint16')


cdef process_dims(sample_sizes, particle_positions, PSF_subpx): # Get resolutions and counts from array dimensions
    subpixels =         PSF_subpx.shape[0]
    psf_res =           PSF_subpx.shape[2]

    sample_count =      sample_sizes.shape[0]

    particle_count =    particle_positions.shape[1]
    step_count =        particle_positions.shape[0]

    return sample_count, particle_count, step_count, subpixels, psf_res


###Convolution functions for specific datatypes

##Float64
ctypedef np.float64_t my_fl64   # Three occurrences
#ctypedef double my_cfl64        # Three occurences
cdef str str_fl64 = "float64"   # One occurence
cdef convolve_fl64(number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_fl64, ndim=1] intensities, np.ndarray[my_fl64, ndim=4] PSF_subpx): ##
    sample_count, particle_count, step_count, subpixels, psf_res = process_dims(sample_sizes, particle_positions, PSF_subpx)

    # Get PSF_subpx indices from particle positions
    PSF_FOV_edge = int(PSF_subpx.shape[2] / 2)
    cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = ((1.0 - (particle_positions - np.floor(particle_positions))) * subpixels - 1.0).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

    # Make arrays stored in C order
    sample_sizes =  np.ascontiguousarray(sample_sizes)
    subpx_poss =    np.ascontiguousarray(subpx_poss)
    offpx_poss =    np.ascontiguousarray(offpx_poss)
    PSF_subpx =     np.ascontiguousarray(PSF_subpx)
    
    cdef np.ndarray[my_fl64, ndim=4] samples = np.zeros([sample_count, step_count, camera_fov_px, camera_fov_px], dtype=str_fl64) ##

    if particle_count != intensities.shape[0]:
        print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
    else:
        cpp_conv_fl64(number_of_threads, sample_count, <my_cfl64*>&samples[0, 0, 0, 0], particle_count, step_count, camera_fov_px, subpixels, psf_res, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <my_cfl64*>&intensities[0], <my_cfl64*>&PSF_subpx[0, 0, 0, 0]) ##

    return samples

##Float32
ctypedef np.float32_t my_fl32   # Three occurrences
#ctypedef float my_cfl32        # Three occurences
cdef str str_fl32 = "float32"   # One occurence
cdef convolve_fl32(number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_fl32, ndim=1] intensities, np.ndarray[my_fl32, ndim=4] PSF_subpx): ##
    sample_count, particle_count, step_count, subpixels, psf_res = process_dims(sample_sizes, particle_positions, PSF_subpx)

    # Get PSF_subpx indices from particle positions
    PSF_FOV_edge = int(psf_res / 2)
    cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = ((1.0 - (particle_positions - np.floor(particle_positions))) * subpixels - 1.0).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

    # Make arrays stored in C order
    sample_sizes =  np.ascontiguousarray(sample_sizes)
    subpx_poss =    np.ascontiguousarray(subpx_poss)
    offpx_poss =    np.ascontiguousarray(offpx_poss)
    PSF_subpx =     np.ascontiguousarray(PSF_subpx)
    
    cdef np.ndarray[my_fl32, ndim=4] samples = np.zeros([sample_count, step_count, camera_fov_px, camera_fov_px], dtype=str_fl32) ##

    if particle_count != intensities.shape[0]:
        print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
    else:
        cpp_conv_fl32(number_of_threads, sample_count, <my_cfl32*>&samples[0, 0, 0, 0], particle_count, step_count, camera_fov_px, subpixels, psf_res, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <my_cfl32*>&intensities[0], <my_cfl32*>&PSF_subpx[0, 0, 0, 0]) ##

    return samples

    
##Int32
ctypedef np.uint32_t my_ui32  # Three occurrences
#ctypedef uint32_t my_cui32   # Three occurences
cdef str str_ui32 = "uint32"  # One occurence
cdef convolve_ui32(number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_ui32, ndim=1] intensities, np.ndarray[my_ui32, ndim=4] PSF_subpx): ##
    sample_count, particle_count, step_count, subpixels, psf_res = process_dims(sample_sizes, particle_positions, PSF_subpx)

    # Get PSF_subpx indices from particle positions
    PSF_FOV_edge = int(PSF_subpx.shape[2] / 2)
    cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = ((1.0 - (particle_positions - np.floor(particle_positions))) * subpixels - 1.0).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

    # Make arrays stored in C order
    sample_sizes =  np.ascontiguousarray(sample_sizes)
    subpx_poss =    np.ascontiguousarray(subpx_poss)
    offpx_poss =    np.ascontiguousarray(offpx_poss)
    PSF_subpx =     np.ascontiguousarray(PSF_subpx)
    
    cdef np.ndarray[my_ui32, ndim=4] samples = np.zeros([sample_count, step_count, camera_fov_px, camera_fov_px], dtype=str_ui32) ##

    if particle_count != intensities.shape[0]:
        print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
    else:
        cpp_conv_ui32(number_of_threads, sample_count, <my_cui32*>&samples[0, 0, 0, 0], particle_count, step_count, camera_fov_px, subpixels, psf_res, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <my_cui32*>&intensities[0], <my_cui32*>&PSF_subpx[0, 0, 0, 0]) ##

    return samples
    
##Int16
ctypedef np.uint16_t my_ui16  # Three occurrences
#ctypedef uint16_t my_cui16   # Three occurences
cdef str str_ui16 = "uint16"  # One occurence
cdef convolve_ui16(number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_ui16, ndim=1] intensities, np.ndarray[my_ui16, ndim=4] PSF_subpx): ##
    sample_count, particle_count, step_count, subpixels, psf_res = process_dims(sample_sizes, particle_positions, PSF_subpx)

    # Get PSF_subpx indices from particle positions
    PSF_FOV_edge = int(PSF_subpx.shape[2] / 2)
    cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = ((1.0 - (particle_positions - np.floor(particle_positions))) * subpixels - 1.0).astype(np.int32)
    cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

    # Make arrays stored in C order
    sample_sizes =  np.ascontiguousarray(sample_sizes)
    subpx_poss =    np.ascontiguousarray(subpx_poss)
    offpx_poss =    np.ascontiguousarray(offpx_poss)
    PSF_subpx =     np.ascontiguousarray(PSF_subpx)
    
    cdef np.ndarray[my_ui16, ndim=4] samples = np.zeros([sample_count, step_count, camera_fov_px, camera_fov_px], dtype=str_ui16) ##

    if particle_count != intensities.shape[0]:
        print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
    else:
        cpp_conv_ui16(number_of_threads, sample_count, <my_cui16*>&samples[0, 0, 0, 0], particle_count, step_count, camera_fov_px, subpixels, psf_res, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <my_cui16*>&intensities[0], <my_cui16*>&PSF_subpx[0, 0, 0, 0]) ##

    return samples




print('Sparse_Subpixel_Convolution module initialized!')