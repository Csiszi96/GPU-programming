#ifndef field
#define field

#include <vector>
#include "gpu.h"
#include "gpu_func.h"

class GPU_Field {
    private:
        // Switches
        bool MOORE = true;
        bool DISTINCT_HOP = true;

        // Variables
        int width;
        int length;
        int size;
        float prec_jump;
        int n_jump;
        float prec_stick;
        int hopping_length;
        float size_shadow;
        int landslide_delta;

        std::vector<int> height;
        std::vector<float> shadow;

        int* p_width;
        int* p_length;
        int* p_size;
        float* p_prec_stick;
        int* p_hopping_length;
        float* p_size_shadow;
        int* p_landslide_delta;
        int* p_height = nullptr;
        float* p_shadow = nullptr;
        int* p_hopping_idxs = nullptr;

        curandState_t* p_States;

        cudaEvent_t evt[2];
        cudaEvent_t tmp_evt;
        cudaStream_t stream;

        float dt = 0.0f;

    public:
        // Overloaded constructors
        GPU_Field();
        GPU_Field(int w, int l);
        GPU_Field(int w, int l, int h);
        GPU_Field(int w, int l, int h, float pj, float prec_stick, int hopping_length, float ss, int ld);

        // Separated becuse of automatic object destruction
        void init(
            int width, int length, int height, 
            float prec_jump, 
            float prec_stick, 
            int hopping_length, 
            float size_shadow, 
            int landslide_delta
        );
        void init(int width, int length, int height);
        void init(int width, int length);

        // Destructor: free up allocated memory on GPU
        ~GPU_Field();

        // Initialize the GPU random generator 
        void init_curand();

        // Simulate the next frame
        float simulate_frame();

        // Get results
        std::vector<int> get_heights();
        std::vector<float> get_shadows();

        void print_field();
        void print_shadows();
};


#endif /* field */