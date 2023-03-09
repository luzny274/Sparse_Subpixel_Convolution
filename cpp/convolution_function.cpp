/*
 * Author: Vitezslav Luzny
 * Contact: luzny274@gmail.com
 * Description: This file contains the implementation of the convolution independent of datatype
*/

#ifndef my_decimal
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

    #define my_decimal double
#endif


inline const int access_poss(const int * ptr, const indint step, const indint p, const indint xy, const indint particle_count){
    return ptr[step * 2 * particle_count + p * 2 + xy];
}
/*
int cpp_conv(   const int number_of_threads,    // Maximum number of threads on which to paralalyze the convolution for loop
                const indint sample_count, 
                my_decimal * samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                const indint particle_count,    
                const indint step_count, 
                const indint camera_fov_px, 
                const indint subpixels, 
                const indint psf_res, 
                const int * subpx_poss,         // Position of particles in subpixel indices
                const int * offpx_poss,         // Position of particles in pixel indices
                const int * sample_sizes,       // Number of particles belonging to a specific sample
                const my_decimal * intensities, // Intensities of individual particles
                const my_decimal * PSF_subpx,   // Point Spread Function array with dimensions (subpixels, subpixels, psf_res, psf_res)
                const int verbose
            ){

    int threads = omp_get_max_threads();

    if (verbose > 0){
        printf("\n");
        printf("OMP max threads: %d\n", threads);
        printf("OMP setting number of threads to %d\n", number_of_threads);
        fflush(stdout);
    }
    
    omp_set_num_threads(number_of_threads);

    auto start = high_resolution_clock::now();

    // Prepare PSF for different threads
    my_decimal** local_PSF_subpxs = new my_decimal*[number_of_threads];

    for (int i = 0; i < number_of_threads; i++){ 
        local_PSF_subpxs[i] = new my_decimal[psf_res * psf_res * subpixels * subpixels];
        memcpy(local_PSF_subpxs[i], PSF_subpx, psf_res * psf_res * subpixels * subpixels * sizeof(my_decimal));
    }

    // Prepare sample indices offsets in particle array
    indint * sample_inds = new indint[sample_count];
    sample_inds[0] = 0;

    for(indint i = 1; i < sample_count; i++)
        sample_inds[i] = sample_inds[i-1] + (indint)sample_sizes[i-1];

    // Check whether array dimensions are valid
    indint particle_count2 = sample_inds[sample_count-1] + (indint)sample_sizes[sample_count-1];
    if (particle_count != particle_count2){
        std::cout << "\nERROR: particle count and sample sizes do not match!" << particle_count << "!=" << particle_count2 << "\n";
        return 0;
    }

    long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    
    if (verbose > 0){
        printf("Preparation of memory took %lld ms\n", duration);
        fflush(stdout);
    }


    start = high_resolution_clock::now();
    // Convolution
    #pragma omp parallel for schedule(dynamic)
    for (indint sample_cursor = 0; sample_cursor < sample_count; sample_cursor++){ // Samples
        indint t_id = omp_get_thread_num(); // Id of current thread
        for (indint s = 0; s < step_count; s++) // Steps (frames)
            for (indint p = sample_inds[sample_cursor]; p < sample_inds[sample_cursor] + (indint)sample_sizes[sample_cursor]; p++) // Particles
            {
                const indint by =   access_poss(offpx_poss, s, p, 0, particle_count); //Offset in main pixel dimension
                const indint bx =   access_poss(offpx_poss, s, p, 1, particle_count); //Offset in main pixel dimension
                const indint suby = access_poss(subpx_poss, s, p, 0, particle_count); //Offset in subpixel dimension
                const indint subx = access_poss(subpx_poss, s, p, 1, particle_count); //Offset in subpixel dimension

                // Dummy variables for constant offsets in arrays
                const indint frame_ind = sample_cursor * camera_fov_px * camera_fov_px * step_count + s * camera_fov_px * camera_fov_px;
                const indint psf_off1_ind = psf_res * psf_res * subpixels;
                const indint psf_off2_ind = psf_res * psf_res;

                // Intensity of the particle
                const my_decimal intensity = intensities[p];

                if(bx > -camera_fov_px && bx < psf_res && by > -camera_fov_px && by < psf_res){ //Check whether bx and by values are valid, skip addition otherwise
                    const indint x1_start = std::max(-by, (indint)0);
                    const indint x2_start = std::max(-bx, (indint)0);
                    const indint x1_end   = std::min(camera_fov_px, x1_start + psf_res - std::max(by, (indint)0)) - x1_start;
                    const indint x2_end   = std::min(camera_fov_px, x2_start + psf_res - std::max(bx, (indint)0)) - x2_start;
                    
                    indint sample_offx1 = frame_ind + (0 + x1_start) * camera_fov_px + x2_start;
                    indint psf_offx1 = suby * psf_off1_ind + subx * psf_off2_ind + (std::max(by, (indint)0) + 0) * psf_res + std::max(bx, (indint)0);
                    for(indint x1 = 0; x1 < x1_end; x1++){
                        my_decimal * samples_off = &samples[sample_offx1];
                        const my_decimal * PSF = &local_PSF_subpxs[t_id][psf_offx1];

                        #pragma omp simd
                        for(indint x2 = 0; x2 < x2_end; x2++)
                            samples_off[x2] += intensity * PSF[x2];

                        sample_offx1 += camera_fov_px;
                        psf_offx1 += psf_res;
                    }
                }
            }
    }

    
    duration = duration_cast<seconds>(high_resolution_clock::now() - start).count();
    
    if (verbose > 0){
        printf("Computation of convolutions took %lld s\n", duration);
        fflush(stdout);
    }

    // Release allocated memory
    for (int i = 0; i < number_of_threads; i++)
        delete[] local_PSF_subpxs[i];
    delete[] local_PSF_subpxs;
    delete[] sample_inds;

    return 1;
}

*/


int cpp_conv(   const int number_of_threads,    // Maximum number of threads on which to paralalyze the convolution for loop
                const indint sample_count, 
                my_decimal * samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                const indint particle_count,    
                const indint step_count, 
                const indint camera_fov_px, 
                const indint subpixels, 
                const indint psf_res, 
                const int * subpx_poss,         // Position of particles in subpixel indices
                const int * offpx_poss,         // Position of particles in pixel indices
                const int * sample_sizes,       // Number of particles belonging to a specific sample
                const my_decimal * intensities, // Intensities of individual particles
                const my_decimal * PSF_subpx    // Point Spread Function array with dimensions (subpixels, subpixels, psf_res, psf_res)
                , const int verbose
            ){

    printf("\n");
    int threads = omp_get_max_threads();
    printf("\nOMP max threads: %d", threads);
    printf("\nOMP setting number of threads to %d", number_of_threads);
    omp_set_num_threads(number_of_threads);

    auto start = high_resolution_clock::now();

    // Prepare PSF for different threads
    my_decimal** local_PSF_subpxs = new my_decimal*[number_of_threads];

    for (int i = 0; i < number_of_threads; i++){ 
        local_PSF_subpxs[i] = new my_decimal[psf_res * psf_res * subpixels * subpixels];
        memcpy(local_PSF_subpxs[i], PSF_subpx, psf_res * psf_res * subpixels * subpixels * sizeof(my_decimal));
    }


    // Prepare sample indices offsets in particle array
    indint * sample_inds = new indint[sample_count];
    sample_inds[0] = 0;

    for(indint i = 1; i < sample_count; i++)
        sample_inds[i] = sample_inds[i-1] + (indint)sample_sizes[i-1];

    // Check whether array dimensions are valid
    indint particle_count2 = sample_inds[sample_count-1] + (indint)sample_sizes[sample_count-1];
    if (particle_count != particle_count2){
        std::cout << "\nERROR: particle count and sample sizes do not match!" << particle_count << "!=" << particle_count2;
        return 0;
    }

    long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    printf("\nPreparation of memory took %lld ms", duration);

    // Convolution
    #pragma omp parallel for schedule(dynamic)
    for (indint sample_cursor = 0; sample_cursor < sample_count; sample_cursor++){ // Samples
        indint t_id = omp_get_thread_num(); // Id of current thread
        for (indint s = 0; s < step_count; s++) // Steps (frames)
            for (indint p = sample_inds[sample_cursor]; p < sample_inds[sample_cursor] + (indint)sample_sizes[sample_cursor]; p++) // Particles
            {
                const indint by =   access_poss(offpx_poss, s, p, 0, particle_count); //Offset in main pixel dimension
                const indint bx =   access_poss(offpx_poss, s, p, 1, particle_count); //Offset in main pixel dimension
                const indint suby = access_poss(subpx_poss, s, p, 0, particle_count); //Offset in subpixel dimension
                const indint subx = access_poss(subpx_poss, s, p, 1, particle_count); //Offset in subpixel dimension

                // Dummy variables for constant offsets in arrays
                const indint frame_ind = sample_cursor * camera_fov_px * camera_fov_px * step_count + s * camera_fov_px * camera_fov_px;
                const indint psf_off1_ind = psf_res * psf_res * subpixels;
                const indint psf_off2_ind = psf_res * psf_res;

                // Intensity of the particle
                const my_decimal intensity = intensities[p];

                if(bx > -camera_fov_px && bx < psf_res && by > -camera_fov_px && by < psf_res){ //Check whether bx and by values are valid, skip addition otherwise
                    const indint x1_start = std::max(-by, (indint)0);
                    const indint x2_start = std::max(-bx, (indint)0);
                    const indint x1_end   = std::min(camera_fov_px, x1_start + psf_res - std::max(by, (indint)0)) - x1_start;
                    const indint x2_end   = std::min(camera_fov_px, x2_start + psf_res - std::max(bx, (indint)0)) - x2_start;
                    
                    indint sample_offx1 = frame_ind + (0 + x1_start) * camera_fov_px + x2_start;
                    indint psf_offx1 = suby * psf_off1_ind + subx * psf_off2_ind + (std::max(by, (indint)0) + 0) * psf_res + std::max(bx, (indint)0);
                    for(indint x1 = 0; x1 < x1_end; x1++){
                        my_decimal * samples_off = &samples[sample_offx1];
                        const my_decimal * PSF = &local_PSF_subpxs[t_id][psf_offx1];

                        #pragma omp simd
                        for(indint x2 = 0; x2 < x2_end; x2++)
                            samples_off[x2] += intensity * PSF[x2];

                        sample_offx1 += camera_fov_px;
                        psf_offx1 += psf_res;
                    }
                }
            }
    }

    // Release allocated memory
    for (int i = 0; i < number_of_threads; i++)
        delete[] local_PSF_subpxs[i];
    delete[] local_PSF_subpxs;
    delete[] sample_inds;

    return 1;
}