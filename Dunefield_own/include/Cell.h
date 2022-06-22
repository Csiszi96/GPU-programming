#include <vector>
#include "mersenne_twister.h"

class Cell {
    private:
        double DEPOSITE_CHANCE_IN_LIGHT = 0.6;
        float SHADOW_LENGTH = 1.5;
        int JUMP_LENGTH = 3;
        int LANDSLIDE_DELTA = 3;

        int height;
        int temp_height;
        bool shadow;
        Random random;

        Cell* forward;
        Cell* backward;
        std::vector<Cell*> sides;
        std::vector<Cell*> back_diags;
        std::vector<Cell*> front_diags;
        std::vector<Cell*> neigbours;
        std::vector<Cell*> diags;

    public:
        Cell(int h);

        // Height or temp height what is stacked or eroded?
        void erode();
        void stack();

        // Define a random float generator
        bool deposited();

        // Probably needs to be rewritten for GPU
        void hop();
        void jump();
        
        void fix_cell();

        int get_height();
        bool get_shadow();

        void set_forward(Cell*);
        void set_backward(Cell*);
        void set_sides(Cell*);
        void set_back_diags(Cell*);
        void set_front_diags(Cell*);

        void set_neigbours();
        void set_diags();

        void landslide(bool moore);

        void calculate_shadow(float init_shadow, int counter);

        void set_shadow_length(float);
        void set_jump_length(int);
        void set_landslide_delta(int);
};