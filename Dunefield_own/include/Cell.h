#include <vector>
#include "random_gen.h"

class Cell {
    private:
        double DEPOSITE_CHANCE_IN_LIGHT = 0.6;
        const int JUMP_LENGTH = 1;
        
        int height;
        int temp_height;
        bool shadow;
        Random random;

        Cell* forward;
        Cell* backward;
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
        void jump(int counter) {
            counter--;
            if (counter == 0) {
                if (!deposited()) {
                    forward->jump(JUMP_LENGTH);
                }
            }
            else {
                forward->jump(counter);
            }
        }
        
        void fix_cell() {
            height = temp_height;
        }
};