/*
 * Author: Vitezslav Luzny
 * Contact: luzny274@gmail.com
 * Description: This file contains the implementation of the convolution independent of datatype
*/

#ifndef my_decimal
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <malloc.h>
    #include <omp.h>
    #include <math.h>

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

inline void * my_alloc(indint size){
    return malloc(size);
}

inline void my_free(void * ptr){
    free(ptr);
}


class ConvolutionCalculator{
    public:
        my_decimal** local_PSF_subpxs = NULL;
        indint psf_res = 0;
        indint subpixels = 0;
        indint optimized_thread_cnt = 0;

        indint async_sample_size = 0;
        my_decimal* async_samples = NULL;
        indint * sample_inds = NULL;

        void prepare_memory(my_decimal* PSF_subpx, // Point Spread Function array with dimensions (subpixels, subpixels, psf_res, psf_res)
                            indint psf_res, 
                            indint subpixels, 
                            indint optimized_thread_cnt,
                            int verbose){
            this->psf_res = psf_res;
            this->subpixels = subpixels;
            this->optimized_thread_cnt = optimized_thread_cnt;

            auto start = high_resolution_clock::now();

            
            if(local_PSF_subpxs != NULL){
                clear_memory();
            }

            // Prepare PSF for different threads
            local_PSF_subpxs = (my_decimal**)my_alloc(optimized_thread_cnt * sizeof(my_decimal*));

            indint psf_sz = psf_res * psf_res * subpixels * subpixels;

            for (int i = 0; i < optimized_thread_cnt; i++){ 
                local_PSF_subpxs[i] = (my_decimal*)my_alloc((psf_sz) * sizeof(my_decimal));
                memcpy(local_PSF_subpxs[i], PSF_subpx, psf_sz * sizeof(my_decimal));
            }
            

            long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
            
            if (verbose > 0){
                printf("Preparation of memory took %lld ms\n", duration);
                fflush(stdout);
            }
        }

        
        int async_convolve(   int number_of_threads,    // Maximum number of threads on which to paralalyze the convolution for loop
                        const indint sample_count,
                        const indint particle_count,    
                        const indint step_count, 
                        const indint camera_fov_px, 
                        const int * subpx_poss,         // Position of particles in subpixel indices
                        const int * offpx_poss,         // Position of particles in pixel indices
                        const int * sample_sizes,       // Number of particles belonging to a specific sample
                        const my_decimal * intensities, // Intensities of individual particles
                        const int verbose){

            if(local_PSF_subpxs == NULL){
                printf("No PSF allocated! Terminating function...\n");
                return 0;
            }

            if(async_samples != NULL){
                printf("There is already a convolution running! Terminating function...\n");
                return 0;
            }

            int threads = omp_get_max_threads();

            if (verbose > 0){
                printf("\n");
                printf("OMP max threads: %d\n", threads);
                printf("OMP setting number of threads to %d\n", number_of_threads);
                fflush(stdout);
            }
            
            omp_set_num_threads(number_of_threads);
            
            async_sample_size = step_count * sample_count * camera_fov_px * camera_fov_px;
            async_samples = (my_decimal*)my_alloc(async_sample_size * sizeof(my_decimal));

            // Prepare sample indices offsets in particle array
            sample_inds = (indint*)my_alloc(sample_count * sizeof(indint));
            sample_inds[0] = 0;

            for(indint i = 1; i < sample_count; i++)
                sample_inds[i] = sample_inds[i-1] + (indint)sample_sizes[i-1];

            // Check whether array dimensions are valid
            indint particle_count2 = sample_inds[sample_count-1] + (indint)sample_sizes[sample_count-1];
            if (particle_count != particle_count2){
                std::cout << "\nERROR: particle count and sample sizes do not match!" << particle_count << "!=" << particle_count2 << "\n";
                return 0;
            }

            // auto start = high_resolution_clock::now();
            
            // Convolution
            
            #pragma omp parallel for schedule(dynamic) nowait
            for (indint sample_cursor = 0; sample_cursor < sample_count; sample_cursor++){ // Samples
                indint t_id = omp_get_thread_num(); // Id of current thread
                t_id %= optimized_thread_cnt;
                
                for (indint s = 0; s < step_count; s++){ // Steps (frames)
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
                                my_decimal * samples_off = &async_samples[sample_offx1];
                                const my_decimal * PSF = &local_PSF_subpxs[t_id][psf_offx1];

                                #pragma omp simd
                                for(indint x2 = 0; x2 < x2_end; x2++){
                                    samples_off[x2] += intensity * PSF[x2];
                                }

                                sample_offx1 += camera_fov_px;
                                psf_offx1 += psf_res;
                            }
                        }
                    }
                }
            }
            // long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
            

            return 1;
        }

        void async_convolve_join(my_decimal * samples, const int verbose){
            if(async_samples == NULL){
                printf("No convolution to be joined!\n");
                return;
            }

            auto start = high_resolution_clock::now();
            #pragma omp taskwait
            long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();

            memcpy(samples, async_samples, sizeof(my_decimal) * async_sample_size);
            my_free(async_samples);
            my_free(sample_inds);
            async_samples = NULL;
            long long duration2 = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();

            if (verbose > 0){
                printf("Waiting for the thread to finish took %lld ms\n", duration);
                printf("Joining processing took %lld ms\n", duration2);
                fflush(stdout);
            }
        }

        void clear_memory(){
            if(local_PSF_subpxs != NULL){
                for (int i = 0; i < optimized_thread_cnt; i++)
                    my_free(local_PSF_subpxs[i]);
                my_free(local_PSF_subpxs);
            }

            local_PSF_subpxs = NULL;
            psf_res = 0;
            subpixels = 0;
            optimized_thread_cnt = 0;
        }
};
