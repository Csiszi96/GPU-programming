
#include "field.h"
#include <random>


Field::Field(int w, int l, int h, float pj, float ps, int hl, float ss) : 
    width(w), length(l), prec_jump(pj), prec_stick(ps), hopping_length(hl), size_shadow(ss) {
    int size = width*length;                // Total number of cells in the field
    height = std::vector<int>(size);        // Height of sand in each cell
    shadow = std::vector<uint8_t>(size);    // Weather the cell is in shadow (vector<bool> did not work, becuse of storage mode)
    n_jump = (int)ceil(size * prec_jump);
    std::vector<int> init_hopping_idxs(n_jump, 0);

    gpu_stream(&stream);

    // NOTE: might do random init
    std::fill(height.begin(), height.end(), h);
    std::fill(shadow.begin(), shadow.end(), false);

    // Allocate memory for each variable
    gpu_malloc((void**) &p_width, sizeof(int));
    gpu_malloc((void**) &p_length, sizeof(int));
    gpu_malloc((void**) &p_size, sizeof(int));
    gpu_malloc((void**) &p_prec_stick, sizeof(float));
    gpu_malloc((void**) &p_hopping_length, sizeof(int));
    gpu_malloc((void**) &p_height, size*sizeof(int));
    gpu_malloc((void**) &p_shadow, size*sizeof(bool));
    gpu_malloc((void**) &p_hopping_idxs, n_jump*sizeof(int));
    gpu_malloc((void**) &p_size_shadow, sizeof(float));

    // Copy variables onto GPU
    std::cout << "1";
    gpu_copy(p_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "2";
    gpu_copy(p_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "3";
    gpu_copy(p_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "4";
    gpu_copy(p_prec_stick, &prec_stick, sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "5";
    gpu_copy(p_hopping_length, &hopping_length, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "6";
    gpu_copy(p_height, height.data(), size*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "7";
    gpu_copy(p_shadow, shadow.data(), size*sizeof(bool), cudaMemcpyHostToDevice);
    std::cout << "8";
    gpu_copy(p_hopping_idxs, init_hopping_idxs.data(), n_jump*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "9";
    gpu_copy(p_size_shadow, &size_shadow, sizeof(float), cudaMemcpyHostToDevice);
    
    init_curand();
    
}

Field::Field(int w, int l, int h) {
    Field::Field(w, l, h, 0.1, 0.6, 5, 1.5);
}

Field::Field(int w, int l) {
    Field::Field(w, l, 3, 0.1, 0.6, 5, 1.5);
}

Field::~Field() {
    cudaEventRecord(evt[0]);

    gpu_free(p_width);
    gpu_free(p_length);
    gpu_free(p_size);
    gpu_free(p_prec_stick);
    gpu_free(p_hopping_length);
    gpu_free(p_height);
    gpu_free(p_shadow);
    gpu_free(p_hopping_idxs);
    gpu_free(p_size_shadow);

    cudaEventRecord(evt[1]);

    cudaEventSynchronize(evt[1]);

    for(auto& e : evt){ cudaEventDestroy(e); }

    std::cout << "Field was safely destroyed." << std::endl;
}

void Field::init_curand(){
    // Initialize the random generator with random values
    std::vector<unsigned long long> Seeds(n_jump);
    std::vector<unsigned long long> Subseqs(n_jump);
    std::vector<unsigned long long> Offsets(n_jump);

    unsigned long long* p_Seeds;
    unsigned long long* p_Subseqs;
    unsigned long long* p_Offsets;

    std::mt19937 mersenne_engine{42};
    std::uniform_int_distribution<unsigned long long> dist{0};

    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(Seeds.begin(),   Seeds.end(),   gen);
    generate(Subseqs.begin(), Subseqs.end(), gen);
    generate(Offsets.begin(), Offsets.end(), gen);

    gpu_malloc((void**) &p_States, n_jump*sizeof(curandState_t));
    gpu_malloc((void**) &p_Seeds, n_jump*sizeof(unsigned long long));
    gpu_malloc((void**) &p_Subseqs, n_jump*sizeof(unsigned long long));
    gpu_malloc((void**) &p_Offsets, n_jump*sizeof(unsigned long long));

    {
        int blockSize = 1;
        int nthreads = n_jump;
        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);
        curand_initialize<<<dimGrid, dimBlock, 0, stream>>>(p_Seeds, p_Subseqs, p_Offsets, p_States);
        gpu_check_err("Problem with initializing curandom: ");
    }

    gpu_free(p_Seeds);
    gpu_free(p_Subseqs);
    gpu_free(p_Offsets);
}

void Field::simulate_frame(int n) {
    cudaEventRecord(evt[0], stream);

    for (int i = 0; i < n; i++) {
        // Generate the new random hoppings
        {
            // NOTE: readjust block size
            int blockSize = 1;
            int nthreads = n_jump;

            dim3 dimGrid (nthreads/blockSize);
            dim3 dimBlock(blockSize);

            curand_generate<<<dimGrid, dimBlock, 0, stream>>>(p_States, p_hopping_idxs, p_size);
            // curand_generate_distinct<<<dimGrid, dimBlock, 0, stream>>>(p_States, p_hopping_idxs, p_size);
            
            gpu_check_err("Problem with generating random states: ");
        }

        // Calculate the shadows on the field 
        {
            // NOTE: readjust block size
            int blockSize = 1;
            int nthreads = width * length;

            dim3 dimGrid (nthreads/blockSize);
            dim3 dimBlock(blockSize);

            shadows<<<dimGrid, dimBlock, 0, stream>>>(p_height, p_shadow, p_length, p_size_shadow);
            gpu_check_err("Problem with initializing curandom: ");
        }

        // Simulate landslides
        {
            // NOTE: readjust block size
            int blockSize = 1;
            int nthreads = width * length;

            dim3 dimGrid (nthreads/blockSize);
            dim3 dimBlock(blockSize);

            landslide<<<dimGrid, dimBlock, 0, stream>>>(p_height, p_length, p_width);
            gpu_check_err("Problem with initializing curandom: ");
        }

        // Get data from GPU
        {
            cudaEventRecord(evt[1], stream); 
            gpu_copy(height.data(), p_height, width * length * sizeof(int), cudaMemcpyDeviceToHost);
            gpu_copy(shadow.data(), p_shadow, width * length * sizeof(bool), cudaMemcpyDeviceToHost);
        }
    }

    cudaEventRecord(evt[1], stream);
    
    cudaEventSynchronize(evt[1]);

    cudaEventElapsedTime(&dt, evt[0], evt[1]);
}

void Field::simulate_frame() {
    simulate_frame(1);
}

std::vector<int> Field::get_heights() {
    return height;
}

std::vector<uint8_t> Field::get_shadows() {
    return shadow;
}