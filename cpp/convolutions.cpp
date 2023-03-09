/*
 * Author: Vitezslav Luzny
 * Contact: luzny274@gmail.com
 * Description: This file imports the convolution function from "conv_func.cpp" to be used with specific datatypes: fl64, fl32, ui32 and ui16
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>



#include <chrono>
using namespace std::chrono;

#define indint int64_t

// include the convolution for a datatype fl64
#define my_decimal double
namespace conv_fl64{
    #include "convolution_function.cpp"
}
void cpp_conv_fl64(const int number_of_threads, const indint sample_count, my_decimal * samples, const indint particle_count, const indint step_count, const indint camera_fov_px, const indint subpixels, const indint psf_res, const int * subpx_poss, const int * offpx_poss, const int * sample_sizes, const my_decimal * intensities, const my_decimal * PSF_subpx, int verbose){
    conv_fl64::cpp_conv(number_of_threads, sample_count, samples, particle_count, step_count, camera_fov_px, subpixels, psf_res, subpx_poss, offpx_poss, sample_sizes, intensities, PSF_subpx, verbose);
}

// include the convolution for a datatype fl32
#undef my_decimal
#define my_decimal float
namespace conv_fl32{
    #include "convolution_function.cpp"
}
void cpp_conv_fl32(const int number_of_threads, const indint sample_count, my_decimal * samples, const indint particle_count, const indint step_count, const indint camera_fov_px, const indint subpixels, const indint psf_res, const int * subpx_poss, const int * offpx_poss, const int * sample_sizes, const my_decimal * intensities, const my_decimal * PSF_subpx, int verbose){
    conv_fl32::cpp_conv(number_of_threads, sample_count, samples, particle_count, step_count, camera_fov_px, subpixels, psf_res, subpx_poss, offpx_poss, sample_sizes, intensities, PSF_subpx, verbose);
}



// include the convolution for a datatype ui32
#undef my_decimal
#define my_decimal uint32_t
namespace conv_ui32{
    #include "convolution_function.cpp"
}
void cpp_conv_ui32(const int number_of_threads, const indint sample_count, my_decimal * samples, const indint particle_count, const indint step_count, const indint camera_fov_px, const indint subpixels, const indint psf_res, const int * subpx_poss, const int * offpx_poss, const int * sample_sizes, const my_decimal * intensities, const my_decimal * PSF_subpx, int verbose){
    conv_ui32::cpp_conv(number_of_threads, sample_count, samples, particle_count, step_count, camera_fov_px, subpixels, psf_res, subpx_poss, offpx_poss, sample_sizes, intensities, PSF_subpx, verbose);
}


// include the convolution for a datatype ui16
#undef my_decimal
#define my_decimal uint16_t
namespace conv_ui16{
    #include "convolution_function.cpp"
}
void cpp_conv_ui16(const int number_of_threads, const indint sample_count, my_decimal * samples, const indint particle_count, const indint step_count, const indint camera_fov_px, const indint subpixels, const indint psf_res, const int * subpx_poss, const int * offpx_poss, const int * sample_sizes, const my_decimal * intensities, const my_decimal * PSF_subpx, int verbose){
    conv_ui16::cpp_conv(number_of_threads, sample_count, samples, particle_count, step_count, camera_fov_px, subpixels, psf_res, subpx_poss, offpx_poss, sample_sizes, intensities, PSF_subpx, verbose);
}