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
    # cdef cppclass ConvolutionCalculator:
    #     void prepare_memory(my_cfl32* PSF_subpx, int64_t psf_res, int64_t subpixels, int64_t optimized_thread_cnt, int verbose)
    #     int async_convolve(int number_of_threads, int64_t sample_count, int64_t particle_count, int64_t step_count, int64_t camera_fov_px, int * subpx_poss, int * offpx_poss, int * sample_sizes, my_cfl32 * intensities, int verbose)
    #     void async_convolve_join(my_cfl32 * samples, int verbose)
    #     void clear_memory()
cdef extern from "../cpp/convolutions.cpp":
    void initialize_conv()


# ctypedef np.float32_t my_fl32   # Three occurrences
# #ctypedef float my_cfl32        # Three occurences
# cdef str str_fl32 = "float32"   # One occurence
# cdef class ConvolutionCalculator_fl32:
#     cdef ConvolutionCalculator conv_calc
#     cdef public int64_t psf_res
#     cdef public int64_t subpixels

#     cdef public int64_t async_sample_count
#     cdef public int64_t async_step_count
#     cdef public int64_t async_camera_fov_px
#     cdef public int64_t async_running


#     def __init__(self, np.ndarray[my_fl32, ndim=4] PSF_subpx, verbose = 0):
#         self.async_running = 0
#         self.subpixels =         PSF_subpx.shape[0]
#         self.psf_res =           PSF_subpx.shape[2]

#         PSF_subpx =              np.ascontiguousarray(PSF_subpx)
#         self.conv_calc.prepare_memory(<my_cfl32*>&PSF_subpx[0, 0, 0, 0], self.psf_res, self.subpixels, 1, verbose)

#     def async_convolve(self, number_of_threads, camera_fov_px, np.ndarray[np.double_t, ndim=3] particle_positions, np.ndarray[np.int32_t, ndim=1] sample_sizes, np.ndarray[my_fl32, ndim=1] intensities, verbose = 0):
#         if self.async_running == 1:
#             print("Thread already running")
#             return

#         sample_count =   sample_sizes.shape[0]
#         particle_count = particle_positions.shape[1]
#         step_count =     particle_positions.shape[0]

#         # Get PSF_subpx indices from particle positions
#         PSF_FOV_edge = int(self.psf_res / 2)
#         cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = (np.ceil((1.0 - (particle_positions - np.floor(particle_positions))) * self.subpixels - 1.0)).astype(np.int32)
#         cdef np.ndarray[np.int32_t, ndim=3] offpx_poss = (PSF_FOV_edge - np.floor(particle_positions) - 1.0).astype(np.int32)    

#         subpx_poss[subpx_poss < 0] = 0
#         subpx_poss[subpx_poss >= self.subpixels] = self.subpixels - 1

#         # Make arrays stored in C order
#         sample_sizes =  np.ascontiguousarray(sample_sizes)
#         subpx_poss =    np.ascontiguousarray(subpx_poss)
#         offpx_poss =    np.ascontiguousarray(offpx_poss)
        
#         if particle_count != intensities.shape[0]:
#             print("ERROR: Particle count in arrays \"particle_positions\" and \"intensities\" do not match")
#         else:
#             self.conv_calc.async_convolve(number_of_threads, sample_count, particle_count, step_count, camera_fov_px, <int*>&subpx_poss[0, 0, 0], <int*>&offpx_poss[0, 0, 0], <int*>&sample_sizes[0], <float*>&intensities[0], verbose) ##

#         self.async_sample_count     = sample_count 
#         self.async_step_count       = step_count   
#         self.async_camera_fov_px    = camera_fov_px
#         self.async_running          = 1

#     def async_convolve_join(self, verbose = 0):
#         cdef np.ndarray[my_fl32, ndim=4] samples = np.zeros([self.async_sample_count, self.async_step_count, self.async_camera_fov_px, self.async_camera_fov_px], dtype=str_fl32) ##
#         if self.async_running == 0:
#             print("No thread to join")
#             return 0

#         self.conv_calc.async_convolve_join(<my_cfl32*>&samples[0, 0, 0, 0], verbose) ##
#         self.async_running          = 0
#         return samples

#     def __del__(self):
#         self.conv_calc.clear_memory()


ctypedef np.float32_t my_fl32   # Three occurrences
#ctypedef float my_cfl32        # Three occurences
cdef str str_fl32 = "float32"   # One occurence
cdef class ConvolutionCalculator_fl32:
    cdef ConvolutionCalculator conv_calc
    cdef public int64_t psf_res
    cdef public int64_t subpixels
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
        cdef np.ndarray[np.int32_t, ndim=3] subpx_poss = (np.ceil((1.0 - (particle_positions - np.floor(particle_positions))) * self.subpixels - 1.0)).astype(np.int32)
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

initialize_conv()