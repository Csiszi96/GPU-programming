#include <vector>
#include <tuple>
#include "Cell.h"
#include "mersenne_twister.h"

class Field {
    private:
        // Module switches
        bool MOORE = true;
        bool DISTINCT_HOP = false;

        // Class variables
        int width;
        int length;
        double hopping_prec;
        Random random;
        // int field_size;
        
        // Might want to change initialization height or randomize (with changable mean)
        int height = 10;
        std::vector<std::vector<Cell*>> field;
        // std::vector<Cell*> field_vector;
        std::vector<Cell*> rnd_field_vector;
        std::vector<Cell*> first_column;

        
    public:
        std::vector<Cell*> field_vector;
        int field_size;

        Field(int w, int l);
        Field(int w, int l, double p);

        // Calculate windshadow
        void calculate_shadows();

        // Select HOPPING_PRECENT of cells and execute hopping
        // Rewrite for no repeating cells
        void hopping();

        // Execute landslides on Cells
        void landslides();

        // Roll the calculated values into the storage
        void fix_cells();

        // Load the pointers of the neighbouring cells into each cell
        void load_neigbours();

        // std::tuple<int,int> warp_coordinates(int x, int y);
        Cell* warp_coordinates(int x, int y);

        // Simulate the next state of the field
        void simulate_frame();

        // Read the state of the sandpit
        std::vector<std::vector<int>> get_heights();
        std::vector<std::vector<bool>> get_shadows();
        std::vector<int> get_heights_arr();
        std::vector<bool> get_shadows_arr();

        // std::vector<std::vector<bool>> get_shadows();
        
        void set_shadow_length(float);
        void set_jump_length(int);
        void set_landslide_delta(int);

        void print_heights();
        void print_shadows();

        int get_width();
};
