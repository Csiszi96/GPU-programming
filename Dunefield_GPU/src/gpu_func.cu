
#include "gpu_func.h"
#include <stdio.h>

__global__ void curand_initialize(unsigned long long* seeds, unsigned long long* subseqs, unsigned long long* offsets, curandState_t* states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seeds[id], subseqs[id], offsets[id], &states[id]);

    // printf("%d %llu %llu %llu %llu\n", id, seeds[id], subseqs[id], offsets[id], &states[id]);
}

__global__ void curand_generate(curandState_t* states, int* p_hopping_idxs, int* size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    curandState localState = states[id];
    float rnd = curand_uniform(&localState);

    float x = ceil(rnd * (*size)) - 1; // as 1.0 is included and 0.0 is excluded
    
    states[id] = localState;
    p_hopping_idxs[id] = (int) x;

    // printf("%d %f\n", id, x);
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

__global__ void hop(int* p_height, int* p_hopping_idxs, float* p_shadow, int* p_length, int* p_width, int* p_hopping_length, float* p_prec_stick, curandState_t* states) {
    // Hopping identifier
    int hop_id = threadIdx.x + blockIdx.x * blockDim.x;

    // Cell identifier that makes the hopping
    unsigned int id = p_hopping_idxs[hop_id];

    // printf("%d %d\n", hop_id, id);

    // Calulating coordinates in the table (NOTE: change to 2d matrix)
    int x = id % *p_length;
    int y_remainder = id - x;

    // Erode original cell
    atomicAdd(&p_height[id], -1);

    // If cell has negative height abort hopping (and stack the original sand block back)
    if (p_height[id] < 0)
        atomicAdd(&p_height[id], 1);

    // Else execute hopping
    else {
        // Save state of random number generator
        curandState localState = states[hop_id];

        // Continue drift until particle is settled
        while(true) {
            // Calculate new x coordinate and id (particle only moves in the x direction)
            x = (x + *p_hopping_length) % *p_length;
            id = y_remainder + x;

            // Check if particle settles (if in shadow or a certain precentage if not)
            if (p_shadow[id] || (curand_uniform(&localState) <= *p_prec_stick)) {
                // If yes, stack it and break hopping
                atomicAdd(&p_height[id], 1);
                break;
            }
        }

        // Save the random generator state
        states[hop_id] = localState;
    }
}

__global__ void shadows(int* p_height,  float* p_shadow, int* p_length, float* size_shadow) {
    int row_id = threadIdx.x + blockIdx.x * blockDim.x;
    int length = *p_length;

    auto id = [&row_id, &length](int x){return (int)(row_id * length + x);};

    auto clac_shadow = [&p_height, &size_shadow](float init_0, int id_0, int id_1) {
        // Calculating overreaching shadow
        float init = (init_0 - 1 < 0) ? 0.0f : (init_0 - 1);
        init = init + (p_height[id_0] - p_height[id_1])* (*size_shadow);

        // printf("%0.2f %d %d %0.2f\n", init_0, p_height[id_0], p_height[id_1], (init < 0) ? 0.0f : (float)init);
        
        return ((init < 0) ? 0.0f : (float)init);
    };

    // Calculate the shadow on the first element
    float init_shadow = 0;
    for (int x = 0; x < length - 1; x++) 
        init_shadow = (float)clac_shadow(init_shadow, id(x), id(x) + 1);

    // The shadow on the first element
    init_shadow = clac_shadow( init_shadow, id(length -1), id(0));


    // Save shadow state of first cell
    p_shadow[row_id * length] = init_shadow;

    // Calculate and save shadows for rest of the cells
    for (int x = 0; x < length - 1; x++) {
        // Shadow of next cell
        init_shadow = (float)clac_shadow(init_shadow, id(x), id(x+1));
        // Save shadow state of next cell
        p_shadow[id(x+1)] = init_shadow;
    }
}

__global__ void landslide(int* p_height, int* p_length, int* p_width, int* p_landslide_delta) {
    // Element id in the field
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Offset in which direction we are calculating the landslide 
    int offset[4][2] = {{0,-1},{-1,0},{0,1},{1,0}};

    auto offset_id = [&offset, &p_length, &p_width](int id, int i){
        int x = id%(*p_length) +  offset[i][0];
        int y = ((int)(id/(*p_length))) + offset[i][1];

        // Warp coordinates
        if (x < 0) x += *p_length;
        if (x >= (*p_length)) x -= *p_length;
        if (y < 0) y += *p_width;
        if (y >= (*p_width)) y -= *p_width;

        int ret_id = y * (*p_length) + x;
        return (int) ret_id;
    };

    // NOTE: how to randomize checks? -> permutation can be described as an integer <= n! - 1
    for (int i = 0; i < 4; i++) {
        int id_2 = offset_id(id, i);
        if ((p_height[id] - p_height[id_2]) >= 3) {
            atomicAdd(&p_height[id], -1);
            atomicAdd(&p_height[id_2], 1);
        } 
    }
}

__global__ void landslide_moore(int* p_height, int* p_length, int* p_width, int* p_landslide_delta) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int offset[4][2] = {{0,-1},{-1,0},{0,1},{1,0}};
    int offset_diag[4][2] = {{-1,-1},{-1,1},{1,1},{1,-1}};
    
    auto offset_id = [&p_length, &p_width](int id, int i, int (*offset)[4][2]) {
            int x = id % (*p_length) +  (*offset)[i][0];
            int y = ((int)(id / (*p_length))) + (*offset)[i][1];

            // Warp coordinates
            if (x < 0) x += *p_length;
            if (x >= (*p_length)) x -= *p_length;
            if (y < 0) y += *p_width;
            if (y >= (*p_width)) y -= *p_width;

            int ret_id = y * (*p_length) + x;
            return (int) ret_id;
        };

    for (int i = 0; i < 4; i++) {
        int id_2 = offset_id(id, i, &offset);
        if ((p_height[id] - p_height[id_2]) >= 3) {
            atomicAdd(&p_height[id], -1);
            atomicAdd(&p_height[id_2], 1);
        } 
    }

    for (int i = 0; i < 4; i++) {
        int id_2 = offset_id(id, i, &offset_diag);
        if ((p_height[id] - p_height[id_2]) >= 3) {
            atomicAdd(&p_height[id], -1);
            atomicAdd(&p_height[id_2], 1);
        } 
    }
}
