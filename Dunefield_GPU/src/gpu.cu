#include "gpu.h"

// template <typename T>
// void gpu_malloc(T src, int size) {
//     cudaError_t err = cudaSuccess;
//     err = cudaMalloc(src , size);
//     if( err != cudaSuccess){ 
//         std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; 
//         exit(-1); 
//     }
// }

// template <typename T1, typename T2>
// void gpu_copy(T1 src, T2 data, int size, cudaMemcpyKind direction) {
//     cudaError_t err = cudaSuccess;
// 	err = cudaMemcpy(src, data, size, direction );
//     if( err != cudaSuccess){
//         if (direction == cudaMemcpyHostToDevice) std::cout << "Error copying memory from host to device: ";
//         if (direction == cudaMemcpyDeviceToHost) std::cout << "Error copying memory from device to host: ";
//         if (direction == cudaMemcpyDeviceToDevice) std::cout << "Error copying memory from device to device: ";
//         if (direction == cudaMemcpyHostToHost) std::cout << "Error copying memory from host to host: ";
        
//         std::cout << cudaGetErrorString(err) << "\n"; 
//         exit(-1); 
//     }
// }

// template <typename T>
// void gpu_free(T src) {
//     cudaError_t err = cudaSuccess;
//     err = cudaFree(src);
//     if( err != cudaSuccess) {
//         std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; 
//         exit(-1); 
//     }
// }

// template <typename T>
// void gpu_free(T src, std::string name) {
//     cudaError_t err = cudaSuccess;
//     err = cudaFree(src);
//     if( err != cudaSuccess) {
//         std::cout 
//             << "Error freeing allocation for " << name << ": " 
//             << cudaGetErrorString(err) << "\n"; 
//         exit(-1); 
//     }
// }


void gpu_check_err(std::string message) {
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if( err != cudaSuccess) {
        std::cout 
            << message << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

void gpu_stream_create(cudaStream_t &stream) {
    cudaError_t err = cudaSuccess;
    err = cudaStreamCreate(&stream);
    if( err != cudaSuccess) {
        std::cout 
            << "Error creating CUDA stream: " << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

void gpu_event_create(cudaEvent_t &event) {
    cudaError_t err = cudaSuccess;
    err = cudaEventCreate(&event);
    if( err != cudaSuccess) {
        std::cout 
            << "Error creating CUDA event: " << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}

void gpu_event(std::string event, cudaEvent_t &evt, cudaStream_t &stream) {
    cudaError_t err = cudaSuccess;
    err = cudaEventRecord(evt, stream);
    if( err != cudaSuccess) {
        std::cout 
            << "Error recording event" << event << ": " << cudaGetErrorString(err) << "\n"; 
        exit(-1); 
    }
}