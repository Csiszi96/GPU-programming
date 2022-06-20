#include <vector>
#include <tuple>
#include "Cell.h"
#include "mersenne_twister.h"

class Field {
    private:
        int width;
        int length;
        double hopping_prec;
        Random random;
        int field_size;
        
        // Might want to change initialization height or randomize (with changable mean)
        int height = 10;
        std::vector<std::vector<Cell*>>* field;
        std::vector<Cell*> field_vector;
        
    public:
        // DONE
        Field(int w, int l);
        Field(int w, int l, double p);

        // Calculate windshadow
        // DONE
        void calculate_shadows();

        // Select HOPPING_PRECENT of cells and execute hopping
        // Rewrite for no repeating cells
        // DONE
        void hopping();

        // Calculate gradients for each Cells
        void calculate_gradients();

        // Execute landslides on Cells
        // DONE
        void landslides();

        // Roll the calculated values into the storage
        // DONE
        void fix_cells();

        // Load the pointers of the neighbouring cells into each cell
        // DONE
        void load_neigbours();

        // std::tuple<int,int> warp_coordinates(int x, int y);
        // DONE
        Cell* warp_coordinates(int x, int y);

        // Simulate the next state of the field
        // DONE
        void simulate_frame();

        // Read the state of the sandpit
        // DONE
        std::vector<std::vector<int>> get_heights();

        // std::vector<std::vector<bool>> get_shadows();
};
