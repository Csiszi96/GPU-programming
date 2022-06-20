// Mersene-twister random generator
#ifndef mt_generator // include guard
#define mt_generator

#include <random>

class Random {
    private:
        std::uniform_real_distribution<double>* dist_uniform_double;
        
        std::random_device rd;
        // std::mt19937 mersenne_twister(112358);
        std::mt19937* mersene_twister;

    public:
        Random() {
            mersene_twister = new std::mt19937(rd());
            dist_uniform_double = new std::uniform_real_distribution<double>(0.0, 1.0);
        }
        
        double real(){
            return (*dist_uniform_double)(*mersene_twister);
        }

        int integer(int a, int b) {
            std::uniform_int_distribution<> dist(a, b);
            return dist(*mersene_twister);
        }
};


#endif /* mt_generator */