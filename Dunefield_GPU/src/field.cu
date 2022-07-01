
#include "field.h"
#include <random>


GPU_Field::GPU_Field(int w, int l, int h, float pj, float ps, int hl, float ss, int ld) {
    std::cout << "Object at:" << this << std::endl;
    init(w, l, h, pj, ps, hl, ss, ld);
}

GPU_Field::GPU_Field(int w, int l, int h) {
    GPU_Field::GPU_Field(w, l, h, 0.1, 0.6, 5, 1.5, 3);
}

GPU_Field::GPU_Field(int w, int l) {
    GPU_Field::GPU_Field(w, l, 3);
}

GPU_Field::GPU_Field() {
    std::cout << "Object at:" << this << std::endl;
}

void GPU_Field::init(int w, int l, int h, float pj, float ps, int hl, float ss, int ld) {
    std::cout << "Init at:" << this << std::endl;

    width = w;
    length = l; 
    prec_jump = pj; 
    prec_stick = ps; 
    hopping_length = hl; 
    size_shadow = ss;
    landslide_delta = ld;

    std::cout << "init object is at:" << this << std::endl;

    size = width*length;                    // Total number of cells in the field
    height = std::vector<int>(size);        // Height of sand in each cell
    // shadow = std::vector<uint8_t>(size);    // Weather the cell is in shadow (vector<bool> did not work, becuse of storage mode)
    shadow = std::vector<float>(size); 
    n_jump = (int)ceil(size * prec_jump);
    std::vector<int> init_hopping_idxs(n_jump, 0);

    std::cout << "Size: " << size << std::endl;

    gpu_stream_create(stream);
    gpu_event_create(evt[0]);
    gpu_event_create(evt[1]);
    gpu_event_create(tmp_evt);

    // NOTE: might do random init
    std::fill(height.begin(), height.end(), h);
    std::fill(shadow.begin(), shadow.end(), 0.0f);

    // height[30] = 16; // NOTE: delete
    // height[99] = 16; // NOTE: delete
    // height[5] = 16; // NOTE: delete

    // Allocate memory for each variable
    std::cout << "CUDA allocate memory... ";
    gpu_malloc((void**) &p_width, sizeof(int));
    gpu_malloc((void**) &p_length, sizeof(int));
    gpu_malloc((void**) &p_size, sizeof(int));
    gpu_malloc((void**) &p_prec_stick, sizeof(float));
    gpu_malloc((void**) &p_hopping_length, sizeof(int));
    gpu_malloc((void**) &p_height, size*sizeof(int));
    gpu_malloc((void**) &p_shadow, size*sizeof(float));
    gpu_malloc((void**) &p_hopping_idxs, n_jump*sizeof(int));
    gpu_malloc((void**) &p_size_shadow, sizeof(float));
    gpu_malloc((void**) &p_landslide_delta, sizeof(int));
    std::cout << "Done..." << std::endl;

    // Copy variables onto GPU
    std::cout << "CUDA copy data to memory... ";
    gpu_copy(p_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_prec_stick, &prec_stick, sizeof(float), cudaMemcpyHostToDevice);
    gpu_copy(p_hopping_length, &hopping_length, sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_height, height.data(), size*sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_shadow, shadow.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    gpu_copy(p_hopping_idxs, init_hopping_idxs.data(), n_jump*sizeof(int), cudaMemcpyHostToDevice);
    gpu_copy(p_size_shadow, &size_shadow, sizeof(float), cudaMemcpyHostToDevice);
    gpu_copy(p_landslide_delta, &landslide_delta, sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Done..." << std::endl;
    
    // Initiate CUDA random generator
    std::cout << "Initiate CUDA random generator... ";
    init_curand();
    std::cout << "Done..." << std::endl;

    
    // gpu_malloc((void**) &p_rnd_nums, n_jump*sizeof(float));
    // gpu_copy(p_rnd_nums, rnd_nums.data(), n_jump*sizeof(float), cudaMemcpyHostToDevice);
}

void GPU_Field::init(int w, int l, int h) {
    init(w, l, h, 0.1, 0.6, 5, 1.5, 3);
}

void GPU_Field::init(int w, int l) {
    init(w, l, 5);
}

GPU_Field::~GPU_Field() {
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
    gpu_free(p_landslide_delta);

    cudaEventRecord(evt[1]);

    cudaEventSynchronize(evt[1]);

    // NOTE: should check for error (add to gpu funcs)
    for(auto& e : evt){ cudaEventDestroy(e); }
    cudaStreamDestroy(stream);

    std::cout << "GPU_Field was safely destroyed." << std::endl;
}

void GPU_Field::init_curand(){

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

    // for (auto x : Seeds) std::cout << x << " "; std::cout << std::endl;
    // for (auto x : Subseqs) std::cout << x << " "; std::cout << std::endl;
    // for (auto x : Offsets) std::cout << x << " "; std::cout << std::endl;

    gpu_malloc((void**) &p_States, n_jump*sizeof(curandState_t));
    gpu_malloc((void**) &p_Seeds, n_jump*sizeof(unsigned long long));
    gpu_malloc((void**) &p_Subseqs, n_jump*sizeof(unsigned long long));
    gpu_malloc((void**) &p_Offsets, n_jump*sizeof(unsigned long long));
// 
    gpu_copy(p_Seeds, Seeds.data(), n_jump*sizeof(unsigned long long), cudaMemcpyHostToDevice);
    gpu_copy(p_Subseqs, Subseqs.data(), n_jump*sizeof(unsigned long long), cudaMemcpyHostToDevice);
    gpu_copy(p_Offsets, Offsets.data(), n_jump*sizeof(unsigned long long), cudaMemcpyHostToDevice);

    {
        int blockSize = 1;
        int nthreads = n_jump;
        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);
        curand_initialize<<<dimGrid, dimBlock, 0, stream>>>(p_Seeds, p_Subseqs, p_Offsets, p_States);
        gpu_check_err("Problem with initializing curandom: ");
    }

    gpu_event("init_curand", tmp_evt, stream);

    gpu_free(p_Seeds);
    gpu_free(p_Subseqs);
    gpu_free(p_Offsets);

    // cudaDeviceSynchronize();
}

float GPU_Field::simulate_frame() {
    // Switches
    bool RND_GEN = true;
    bool HOPPING = true;
    bool SHADOWS = false;
    bool LANDSLIDE = false;

    cudaEventRecord(evt[0], stream);
    
    // Generate the new random hoppings
    if(RND_GEN) {
        // NOTE: readjust block size
        int blockSize = 1;
        int nthreads = n_jump;

        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);

        if(DISTINCT_HOP) curand_generate_distinct<<<dimGrid, dimBlock, 0, stream>>>(p_States, p_hopping_idxs, p_size);
        else curand_generate<<<dimGrid, dimBlock, 0, stream>>>(p_States, p_hopping_idxs, p_size);
        
        gpu_check_err("Problem with generating random states: ");
    }

    // Simulate the drift of random particles in the wind
    if(HOPPING) {
        // NOTE: readjust block size
        int blockSize = 1;
        int nthreads = n_jump;

        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);

        hop<<<dimGrid, dimBlock, 0, stream>>>(
            p_height, 
            p_hopping_idxs, 
            p_shadow, 
            p_length, 
            p_width,
            p_hopping_length,
            p_prec_stick,
            p_States
        );

        gpu_check_err("Problem with simulating hoppings: ");
    }
    
    // Calculate the shadows on the field 
    if(SHADOWS) {
        // NOTE: readjust block size
        int blockSize = 1;
        int nthreads = width;

        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);

        shadows<<<dimGrid, dimBlock, 0, stream>>>(p_height, p_shadow, p_length, p_size_shadow);
        gpu_check_err("Problem with initializing curandom: ");
    }

    // Simulate landslides 
    if(LANDSLIDE) {
        // NOTE: readjust block size
        int blockSize = 1;
        int nthreads = width * length;

        dim3 dimGrid (nthreads/blockSize);
        dim3 dimBlock(blockSize);

        if(MOORE) landslide_moore<<<dimGrid, dimBlock, 0, stream>>>(p_height, p_length, p_width, p_landslide_delta);
        else landslide<<<dimGrid, dimBlock, 0, stream>>>(p_height, p_length, p_width, p_landslide_delta);

        gpu_check_err("Problem with initializing curandom: ");
    }

     cudaEventRecord(evt[1], stream);

    // Extract variables from GPU
    gpu_copy(height.data(), p_height, size * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_copy(shadow.data(), p_shadow, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Sincronize with CPU and record calculation time
    cudaEventRecord(tmp_evt, stream);
    cudaEventSynchronize(tmp_evt);
    cudaEventElapsedTime(&dt, evt[0], evt[1]);

    return dt;
}

std::vector<int> GPU_Field::get_heights() {
    return height;
}

std::vector<float> GPU_Field::get_shadows() {
    // return std::vector<bool> (shadow.begin(), shadow.end());
    return shadow;
}

void GPU_Field::print_field() {
    bool with_shadow = false;
    for (int y = 0; y < width; y ++) {
        for (int x = 0; x < length; x++) {
            std::cout << height[y * width + x];
            if(with_shadow) std::cout << ":" << shadow[y * width + x];
            std::cout << "\t";
        }
        std::cout << std::endl;
    }
}

void GPU_Field::print_shadows() {
    for (int y = 0; y < width; y ++) {
        for (int x = 0; x < length; x++)
            std::cout << (bool)shadow[y * width + x] << "\t";
        std::cout << std::endl;
    }
}
