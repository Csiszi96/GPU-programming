// Mersene-twister random generator
#ifndef mersenne_twister // include guard
#define mersenne_twister

#include <random>

class Random {
    private:
        std::random_device rd;
        std::mt19937_64 mersenne_twister(rd());
        // std::mt19937 mersenne_twister(112358);

        std::uniform_real_distribution<> dist_uniform_double(0.0, 1.0);

    public:
        Random() {}
        
        double real(){
            return dist_uniform_double(mersenne_twister);
        }

        int integer(int a, int b) {
            std::uniform_int_distribution<> dist(a, b);
            return dist(mersenne_twister);
        }
};

#endif /* mersenne_twister */