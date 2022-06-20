#ifndef gpu_functions
#define gpu_functions

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void curand_initialize(
    unsigned long long* seeds, 
    unsigned long long* subseqs, 
    unsigned long long* offsets, 
    curandState_t* states
);

// Generate a set of coordinates to hop (one version with distinct hopping places)
__global__ void curand_generate(curandState_t* states, int* p_hopping_idxs, int* p_size);
__global__ void curand_generate_distinct(curandState_t* states, int* p_hopping_idxs, int* size);

// Execute hops
__global__ void hop(
    int* p_height, 
    int* p_hopping_idxs, 
    int* p_shadow, 
    int* p_length, 
    int* p_width, 
    int* p_hopping_length, 
    float* p_prec_stick, 
    curandState_t* states
);

// Calculate shadows
__global__ void shadows(int* p_height,  bool* p_shadow, int* p_length, float* p_size_shadow);

// Simulate landslides
__global__ void landslide(int* p_height, int* p_length, int* p_width);
__global__ void landslide_moore(int* p_height, int* p_length, int* p_width);


#endif /* gpu_functions */