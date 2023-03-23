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
    // #include <omp.h>
    #include <math.h>
    #include <future>

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

inline void debug_log(const char * txt){
    FILE * f = fopen("log.txt", "a");
    fprintf(f, "%s\n", txt);
    fclose(f);
}


class ConvolutionCalculator{
    public:
        my_decimal* local_PSF_subpx = NULL;
        indint psf_res = 0;
        indint subpixels = 0;

        my_decimal * async_samples = NULL;
        indint async_sample_size = 0;
        indint * async_sample_inds = NULL;
        std::vector<std::thread> async_threads;

        void prepare_memory(my_decimal* PSF_subpx, // Point Spread Function array with dimensions (subpixels, subpixels, psf_res, psf_res)
                            indint psf_res, 
                            indint subpixels, 
                            int verbose){
            this->psf_res = psf_res;
            this->subpixels = subpixels;

            auto start = high_resolution_clock::now();

            
            if(local_PSF_subpx != NULL){
                clear_memory();
            }

            // Prepare PSF
            indint psf_sz = psf_res * psf_res * subpixels * subpixels;

            local_PSF_subpx = (my_decimal*)my_alloc((psf_sz) * sizeof(my_decimal));
            memcpy(local_PSF_subpx, PSF_subpx, psf_sz * sizeof(my_decimal));
            

            long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
            
            if (verbose > 0){
                printf("Preparation of memory took %lld ms\n", duration);
                fflush(stdout);
            }
        }

        static int static_convolve(const indint sample_count, 
                        my_decimal * samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                        const indint particle_count,    
                        const indint step_count, 
                        const indint camera_fov_px, 
                        const int * subpx_poss,         // Position of particles in subpixel indices
                        const int * offpx_poss,         // Position of particles in pixel indices
                        const int * sample_sizes,       // Number of particles belonging to a specific sample
                        const my_decimal * intensities, // Intensities of individual particles
                        const int verbose,
                        const my_decimal * PSF_subpxs,
                        const indint psf_res,
                        const indint subpixels,
                        const indint sample_begin,
                        const indint sample_end,
                        const indint * sample_inds){

            if(PSF_subpxs == NULL){
                printf("No PSF allocated! Terminating function...\n");
                return 0;
            }
            

            

            auto start = high_resolution_clock::now();

            
            // Convolution
            for (indint sample_cursor = sample_begin; sample_cursor < sample_end; sample_cursor++){ // Samples
                
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
                                my_decimal * samples_off = &samples[sample_offx1];
                                const my_decimal * PSF = &PSF_subpxs[psf_offx1];

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
            long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
            
            if (verbose > 0){
                printf("Computation of convolutions took %lld ms\n", duration);
                fflush(stdout);
            }


            return 1;
        }

        
        int convolve(   int number_of_threads,    // Maximum number of threads on which to paralalyze the convolution for loop
                        const indint sample_count, 
                        my_decimal * samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                        const indint particle_count,    
                        const indint step_count, 
                        const indint camera_fov_px, 
                        const int * subpx_poss,         // Position of particles in subpixel indices
                        const int * offpx_poss,         // Position of particles in pixel indices
                        const int * sample_sizes,       // Number of particles belonging to a specific sample
                        const my_decimal * intensities, // Intensities of individual particles
                        const int verbose){

            

            // Prepare sample indices offsets in particle array
            indint * sample_inds = (indint*)my_alloc(sample_count * sizeof(indint));
            sample_inds[0] = 0;

            for(indint i = 1; i < sample_count; i++)
                sample_inds[i] = sample_inds[i-1] + (indint)sample_sizes[i-1];

            // Check whether array dimensions are valid
            indint particle_count2 = sample_inds[sample_count-1] + (indint)sample_sizes[sample_count-1];
            if (particle_count != particle_count2){
                std::cout << "\nERROR: particle count and sample sizes do not match!" << particle_count << "!=" << particle_count2 << "\n";
                return 0;
            }

            
            int val = static_convolve(sample_count, 
                                   samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                                   particle_count,    
                                   step_count, 
                                   camera_fov_px, 
                                   subpx_poss,         // Position of particles in subpixel indices
                                   offpx_poss,         // Position of particles in pixel indices
                                   sample_sizes,       // Number of particles belonging to a specific sample
                                   intensities, // Intensities of individual particles
                                   verbose,
                                   local_PSF_subpx,
                                   psf_res,
                                   subpixels,
                                   0,
                                   sample_count,
                                   sample_inds);
                                   
            return val;
        }
        

        void async_convolve(int number_of_threads,    // Maximum number of threads on which to paralalyze the convolution for loop
                        const indint sample_count, 
                        const indint particle_count,    
                        const indint step_count, 
                        const indint camera_fov_px, 
                        const int * subpx_poss,         // Position of particles in subpixel indices
                        const int * offpx_poss,         // Position of particles in pixel indices
                        const int * sample_sizes,       // Number of particles belonging to a specific sample
                        const my_decimal * intensities, // Intensities of individual particles
                        const int verbose){
            if(async_samples != NULL || !async_threads.empty()){
                printf("Last convolution didn't finish!!\n");
                return;
            }

            async_sample_size = step_count * sample_count * camera_fov_px * camera_fov_px;
            async_samples = (my_decimal*)my_alloc(async_sample_size * sizeof(my_decimal));

            // Prepare sample indices offsets in particle array
            indint * async_sample_inds = (indint*)my_alloc(sample_count * sizeof(indint));
            async_sample_inds[0] = 0;

            for(indint i = 1; i < sample_count; i++)
                async_sample_inds[i] = async_sample_inds[i-1] + (indint)sample_sizes[i-1];

            // Check whether array dimensions are valid
            indint particle_count2 = async_sample_inds[sample_count-1] + (indint)sample_sizes[sample_count-1];
            if (particle_count != particle_count2){
                std::cout << "\nERROR: particle count and sample sizes do not match!" << particle_count << "!=" << particle_count2 << "\n";
                return;
            }

            indint particles_per_thread = particle_count / number_of_threads;
            indint last_sample = 0;

            char txt[128];
            for(int i = 0; i < number_of_threads; i++){
                // indint sample_begin = last_sample;
                // indint cur_particles = 0;
                // for(last_sample = last_sample; cur_particles < particles_per_thread && last_sample < sample_count; last_sample++)
                //     cur_particles += sample_sizes[last_sample];

                // sprintf(txt, "%ld", last_sample);
                // debug_log(txt);

                std::thread t(static_convolve, 
                                                    sample_count, 
                                                    async_samples,           // Output array with dimensions (sample_count, step_count, camera_fov_px, camera_fov_px)
                                                    particle_count,    
                                                    step_count, 
                                                    camera_fov_px, 
                                                    subpx_poss,         // Position of particles in subpixel indices
                                                    offpx_poss,         // Position of particles in pixel indices
                                                    sample_sizes,       // Number of particles belonging to a specific sample
                                                    intensities, // Intensities of individual particles
                                                    verbose,
                                                    local_PSF_subpx,
                                                    psf_res,
                                                    subpixels,
                                                    0,
                                                    sample_count,
                                                    async_sample_inds);

                // async_threads.push_back(std::move(t));
                // std::this_thread::sleep_for(45s);

                t.join();

                debug_log("huhu");
            }
            

        }

        void async_convolve_join(my_decimal * samples, const int verbose){
            if(async_samples == NULL){
                printf("No convolution to be joined!\n");
                return;
            }

            auto start = high_resolution_clock::now();
            for(std::thread& t : async_threads)
                t.join();
            async_threads.clear();
            long long duration = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();

            memcpy(samples, async_samples, sizeof(my_decimal) * async_sample_size);
            my_free(async_samples);
            my_free(async_sample_inds);
            async_samples = NULL;
            long long duration2 = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();

            if (verbose > 0){
                printf("Waiting for the thread to finish took %lld ms\n", duration);
                printf("Joining processing took %lld ms\n", duration2);
                fflush(stdout);
            }
        }

        void clear_memory(){
            if(local_PSF_subpx != NULL){
                my_free(local_PSF_subpx);
            }

            local_PSF_subpx = NULL;
            psf_res = 0;
            subpixels = 0;
        }
};
