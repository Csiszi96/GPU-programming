#ifndef cuda_functions
#define cuda_functions

#include <iostream>

// template <typename T>
// void gpu_malloc(T src, int size);

// template <typename T1, typename T2>
// void gpu_copy(T1 src, T2 data, int size, cudaMemcpyKind direction);

// template <typename T>
// void gpu_free(T src);

// template <typename T>
// void gpu_free(T src, std::string name);

void gpu_check_err(std::string message);

void gpu_stream(cudaStream_t* stream);

/////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void gpu_malloc(T src, int size) {
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(src , size);
    if( err != cudaSuccess){ 
        std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

template <typename T1, typename T2>
void gpu_copy(T1 src, T2 data, size_t size, cudaMemcpyKind direction) {
    cudaError_t err = cudaSuccess;
	err = cudaMemcpy(src, data, size, direction);
    if( err != cudaSuccess){
        if (direction == cudaMemcpyHostToDevice) std::cout << "Error copying memory from host to device: ";
        if (direction == cudaMemcpyDeviceToHost) std::cout << "Error copying memory from device to host: ";
        if (direction == cudaMemcpyDeviceToDevice) std::cout << "Error copying memory from device to device: ";
        if (direction == cudaMemcpyHostToHost) std::cout << "Error copying memory from host to host: ";
        
        std::cout << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

template <typename T>
void gpu_free(T src) {
    cudaError_t err = cudaSuccess;
    err = cudaFree(src);
    if( err != cudaSuccess) {
        std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

template <typename T>
void gpu_free(T src, std::string name) {
    cudaError_t err = cudaSuccess;
    err = cudaFree(src);
    if( err != cudaSuccess) {
        std::cout 
            << "Error freeing allocation for " << name << ": " 
            << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

#endif /* cuda_functions */