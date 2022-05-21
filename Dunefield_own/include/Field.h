#include <vector>
#include "Cell.h"

class Field {
    private:
        int width;
        int length;
        double hopping_prec;
        Random random;
        
        // Might want to change initialization height or randomize (with changable mean)
        int height = 10;
        std::vector<std::vector<Cell*>> field(length, std::vector<Cell*>(width, new Cell(height)));
        std::vector<Cell*> field_vector;
        
    public:
        // DONE
        Field(int w, int l);
        Field(int w, int l, double p);

        // Calculate windshadow
        void calculate_sadows();

        // Select HOPPING_PRECENT of cells and execute hopping
        // Rewrite for no repeating cells
        void hopping();

        // Calculate gradients for each Cells
        void calculate_gradients();

        // Execute landslides on Cells
        void landslide();

        // Roll the calculated values into the storage
        // DONE
        void fix_cells();

        // Simulate the next state of the field
        // DONE
        void simulate_step();

        // Load the pointers of the neighbouring cells into each cell
        void load_neigbours();
        arr[2] int warp_coordinates(int x, int y);
};
