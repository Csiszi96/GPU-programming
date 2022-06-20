#include <vector>
#include "mersenne_twister.h"

class Cell {
    private:
        double DEPOSITE_CHANCE_IN_LIGHT = 0.6;
        float SHADOW_LENGTH = 1.5;
        int JUMP_LENGTH = 1;
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

    public:
        // DONE
        Cell(int h);

        // Height or temp height what is stacked or eroded?
        // DONE
        void erode();
        void stack();

        // Define a random float generator
        // DONE
        bool deposited();

        // Probably needs to be rewritten for GPU
        // DONE
        void jump();
        void jump(int counter);
        
        // DONE
        void fix_cell();

        // DONE
        int get_height();

        // DONE
        void set_forward(Cell*);
        void set_backward(Cell*);
        void set_sides(Cell*);
        void set_back_diags(Cell*);
        void set_front_diags(Cell*);

        // DONE
        void set_neigbours();

        void landslide();

        void calculate_shadow(float init_shadow, int counter);
};