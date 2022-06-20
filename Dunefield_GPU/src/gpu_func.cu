
#include "gpu_func.h"

__global__ void curand_initialize(unsigned long long* seeds, unsigned long long* subseqs, unsigned long long* offsets, curandState_t* states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seeds[id], subseqs[id], offsets[id], &states[id]);
}

__global__ void curand_generate(curandState_t* states, int* p_hopping_idxs, int* size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[id];
    float x = ceil(curand_uniform(&localState) * (*size)) - 1; // as 1.0 is included and 0.0 is excluded
    
    states[id] = localState;
    p_hopping_idxs[id] = (int) x;
}

__global__ void curand_generate_distinct(curandState_t* states, int* p_hopping_idxs, int* size){
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[id];
    float x = ceil(curand_uniform(&localState) * ((*size) - id)) - 1;
    
    states[id] = localState;
    p_hopping_idxs[id] = (int) x;

    __syncthreads();

    int offset = 0;
    if (id != 0) {
        for (int i = 0; i < id; i++){
            if (p_hopping_idxs[i] <= p_hopping_idxs[id]){
                offset++;
            }
        }
    }
    __syncthreads();

    p_hopping_idxs[id] += offset;
}

__global__ void hop(int* p_height, int* p_hopping_idxs, int* p_shadow, int* p_length, int* p_width, int* p_hopping_length, float* p_prec_stick, curandState_t* states) {
    int hop_id = threadIdx.x + blockIdx.x * blockDim.x;
    int id = p_hopping_idxs[hop_id];

    int x = id % *p_length;
    int y_remainder = id - x;

    int id_2 = 0;

    atomicAdd(&p_hopping_idxs[id], -1);
    if (p_hopping_idxs[id] < 0)
        atomicAdd(&p_hopping_idxs[id], 1);
    else {
        curandState localState = states[hop_id];
        for(bool cont = true; cont;) {
            x = (x + *p_hopping_length) % *p_length;
            id_2 = y_remainder + x;
            if (p_shadow[id_2] || (curand_uniform(&localState) <= *p_prec_stick)) {
                atomicAdd(&p_hopping_idxs[id], 1);
                cont = false;
            }
        }
        states[hop_id] = localState;
    }
}

__global__ void shadows(int* p_height,  bool* p_shadow, int* p_length, float* size_shadow) {
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate the shaow on the first element
    float init_shadow = 0;
    int id;
    for (int x = 0; x < *p_length - 1; x++) {
        id =  row_id * (*p_length) + x;
        init_shadow = init_shadow - 1 + (p_height[id] - p_height[id + 1])* (*size_shadow);
        if (init_shadow < 0) init_shadow = 0;
    }

    init_shadow = init_shadow - 1 + (
            p_height[(row_id + 1) * (*p_length) - 1] -    // last element of the row
            p_height[row_id * (*p_length)]                // first element of the row
        )* (*size_shadow);
    if (init_shadow < 0) init_shadow = 0;

    // Calculate and save shadows
    for (int x = 0; x < *p_length - 1; x++) {
        if (init_shadow <= 0) {
            p_shadow[id] = false;
            init_shadow = 0;
        }
        else 
            p_shadow[id] = true;
        int id =  row_id * (*p_length) + x;
        init_shadow = init_shadow - 1 + (p_height[id] - p_height[id + 1])* (*size_shadow);
    }
    p_shadow[(row_id + 1) * (*p_length) - 1] = (init_shadow <= 0) ? false : true;
}

__global__ void landslide(int* p_height, int* p_length, int* p_width) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int offset[4][2] = {{0,-1},{-1,0},{0,1},{1,0}};

    // NOTE: how to randomize checks? 
    for (int i = 0; i < 4; i++) {
        int x = offset[i][0];
        int y = offset[i][1];

        // Warp coordinates
        if (x < 0) x += *p_length;
        if (x >= (*p_length)) x -= *p_length;
        if (y < 0) x += *p_width;
        if (y >= (*p_width)) x -= *p_width;

        int id_2 = y * (*p_length) + x;
        if ((p_height[id_2] - p_height[id]) > 2) {
            atomicAdd(&p_height[id], 1);
            atomicAdd(&p_height[id_2], -1);
        } 
    }
}

__global__ void landslide_moore(int* p_height, int* p_length, int* p_width) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int offset[4][2] = {{0,-1},{-1,0},{0,1},{1,0}};
    int offset_2[4][2] = {{-1,-1},{-1,1},{1,1},{1,-1}};

    for (int i = 0; i < 4; i++) {
        int x = offset[i][0];
        int y = offset[i][1];

        // Warp coordinates
        if (x < 0) x += *p_length;
        if (x >= (*p_length)) x -= *p_length;
        if (y < 0) x += *p_width;
        if (y >= (*p_width)) x -= *p_width;

        int id_2 = y * (*p_length) + x;
        if ((p_height[id_2] - p_height[id]) > 2) {
            atomicAdd(&p_height[id], 1);
            atomicAdd(&p_height[id_2], -1);
        } 
    }

    for (int i = 0; i < 4; i++) {
        int x = offset_2[i][0];
        int y = offset_2[i][1];

        // Warp coordinates
        if (x < 0) x += *p_length;
        if (x >= (*p_length)) x -= *p_length;
        if (y < 0) x += *p_width;
        if (y >= (*p_width)) x -= *p_width;

        int id_2 = y * (*p_length) + x;
        if ((p_height[id_2] - p_height[id]) > 2) {
            atomicAdd(&p_height[id], 1);
            atomicAdd(&p_height[id_2], -1);
        } 
    }
}
