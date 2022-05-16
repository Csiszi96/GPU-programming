#include <vector>
#include "Cell.h"

class Field {
    private:
        int width;
        int length;
        
        // Might want to change initialization height
        int height = 10;
        std::vector<std::vector<Cell*>> field(length, std::vector<Cell*>(width, new Cell(height)));
        
    public:
        Field(int width, int length);
        ~Field();
        void calculate_sadows();
        void hopping();
        void calculate_gradients();
        void landslide();
        void simulate_step();
};