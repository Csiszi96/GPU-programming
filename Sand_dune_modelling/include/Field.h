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
        Field(int width, int length) {
            this->width = width;
            this->length = length;
        };
        void calculate_sadows();
        void hopping(); // Select HOPPING_PRECENT of cells and execute hopping
        void calculate_gradients(); // Calculate gradients for each Cells
        void landslide(); // Execute landlides on Cells

        void simulate_step() {
            calculate_sadows();
            hopping();
            calculate_gradients();
            landslide();
        }
};