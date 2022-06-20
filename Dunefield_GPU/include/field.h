#ifndef field
#define field

#include <vector>
#include "gpu.h"
#include "gpu_func.h"

class Field {
    private:
        int width;
        int length;
        float prec_jump;
        int n_jump;
        float prec_stick;
        int hopping_length;
        float size_shadow;

        std::vector<int> height;
        std::vector<uint8_t> shadow;

        int* p_width;
        int* p_length;
        int* p_size;
        float* p_prec_stick;
        int* p_hopping_length;
        int* p_height = nullptr;
        bool* p_shadow = nullptr;
        int* p_hopping_idxs = nullptr;
        float* p_size_shadow;

        curandState_t* p_States;

        cudaEvent_t evt[2];
        cudaStream_t stream;

        float dt = 0.0f;

    public:
        // Overloaded constructors
        Field(int w, int l);
        Field(int w, int l, int h);
        Field(int w, int l, int h, float pj, float prec_stick, int hopping_length, float ss);


        // Destructor: free up allocated memory on GPU
        ~Field();

        // Initialize the GPU random generator 
        void init_curand();

        // Simulate the next frame
        void simulate_frame();
        void simulate_frame(int n);

        // Get results
        std::vector<int> get_heights();
        std::vector<uint8_t> get_shadows();

};


#endif /* field */