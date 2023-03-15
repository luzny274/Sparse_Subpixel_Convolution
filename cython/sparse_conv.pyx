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

from cython.operator cimport dereference as deref


# Import the convolution functions from c++ 
ctypedef double   my_cfl64
ctypedef float    my_cfl32
ctypedef uint32_t my_cui32 
ctypedef uint16_t my_cui16 

cdef extern from "../cpp/convolutions.cpp" namespace "conv_fl32":
    cdef cppclass ConvolutionCalculator:
        void prepare_memory(my_cfl32* PSF_subpx, int64_t psf_res, int64_t subpixels, int64_t optimized_thread_cnt, int verbose)
        int convolve(int number_of_threads, int64_t sample_count, my_cfl32 * samples, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cfl32 * intensities, int verbose)
        void clear_memory()


eps = 1e-10

ctypedef np.float32_t my_fl32   # Three occurrences
#ctypedef float my_cfl32        # Three occurences
cdef str str_fl32 = "float32"   # One occurence
cdef class ConvolutionCalculator_fl32:
    cdef ConvolutionCalculator conv_calc
    psf_res = 0
    subpixels = 0
    def __init__(self, np.ndarray[my_fl32, ndim=4] PSF_subpx, optimized_thread_cnt, verbose = 0):
        self.subpixels =         PSF_subpx.shape[0]
        self.psf_res =           PSF_subpx.shape[2]

        PSF_subpx =              np.ascontiguousarray(PSF_subpx)
        self.conv_calc.prepare_memory(<my_cfl32*>&PSF_subpx[0, 0, 0, 0], self.psf_res, self.subpixels, optimized_thread_cnt, verbose)

    def convolve(self, number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_fl32, ndim=1] intensities, verbose = 0):
        sample_count =   sample_sizes.shape[0]
        particle_count = particle_positions.shape[1]
        step_count =     particle_positions.shape[0]

        # Get PSF_subpx indices from particle positions
        PSF_FOV_edge = int(self.psf_res / 2)
        cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = ((1.0 - (particle_positions - np.floor(particle_positions))) * self.subpixels - eps).astype(np.int32)
        cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

        subpx_poss[subpx_poss < 0] = 0
        subpx_poss[subpx_poss >= self.subpixels] = self.subpixels - 1

        # Make arrays stored in C order
        sample_sizes =  np.ascontiguousarray(sample_sizes)
        subpx_poss =    np.ascontiguousarray(subpx_poss)
        offpx_poss =    np.ascontiguousarray(offpx_poss)
        
        cdef np.ndarray[my_fl32, ndim=4] samples = np.zeros([sample_count, step_count, camera_fov_px, camera_fov_px], dtype=str_fl32) ##

        if particle_count != intensities.shape[0]:
            print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
        else:
            self.conv_calc.convolve(number_of_threads, sample_count, <my_cfl32*>&samples[0, 0, 0, 0], particle_count, step_count, camera_fov_px, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <float*>&intensities[0], verbose) ##
        return samples

    def __del__(self):
        self.conv_calc.clear_memory()



#def convolve(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, kernel, datatype, verbose=1):
#    """
#    This function takes seven arguments:
#
#    number_of_threads : int
#        Maximum number of threads on which to paralalyze the convolution for loop
#    camera_fov_px : int
#        Dimension of the output
#    particle_positions : numpy.ndarray[np.double_t, ndim=3]
#        Array of individual particle positions in different frames, dimensions: (step_count, particle_count, 2 coordinates)
#    sample_sizes : numpy.ndarray[np.int32_t, ndim=1]
#        Number of particles belonging to a specific sample, dimension: (particle_count)
#    intensities : numpy.ndarray[datatype, ndim=1]
#        Intensities of individual particles, dimension: (particle_count)
#    kernel : numpy.ndarray[datatype, ndim=4]
#        Point Spread Function array, dimensions: (subpixels, subpixels, psf_res, psf_res)
#    datatype : type
#        datatype of intensities, PSF and output. Possible values: numpy.float64, numpy.float32, numpy.uint32, numpy.uint16
#    verbose : int
#        0 - print only errors, above 0 - print more
#
#    Returns
#    -------
#    numpy.ndarray[datatype, ndim=4]
#        Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px). If convolution could not happen due to invalid input, it is filled with zeros.
#    """
#
#    if not isinstance(number_of_threads, int):
#        raise TypeError('number_of_threads must be an integer')
#    if not isinstance(camera_fov_px, int):
#        raise TypeError('camera_fov_px must be an integer')
#
#    if datatype == np.float64:
#        return convolve_fl64(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, kernel, verbose)
#    elif datatype == np.float32:
#        return convolve_fl32(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, kernel, verbose)
#    elif datatype == np.uint32:
#        return convolve_ui32(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, kernel, verbose)
#    elif datatype == np.uint16:
#        return convolve_ui16(number_of_threads, camera_fov_px, particle_positions, sample_sizes, intensities, kernel, verbose)
#    else:
#        raise TypeError('Datatype argument must be either numpy.float64, numpy.float32, numpy.uint32 or numpy.uint16')

