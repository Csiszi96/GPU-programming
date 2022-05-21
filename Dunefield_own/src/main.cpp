#include "Field.h"
#include <random>

int const WIDTH = 512;
int const LENGTH = 512;
const int JUMP_LENGTH = 1;
const float DEPOSITE_CHANCE_LIGHT = 0.6;
const float HOPPING_PRECENT = 0.1;

// const float SHADOW_ANGLE = 15;

std::random_device rd;
// std::mt19937 mersenne_twister(rd());
std::mt19937 mersenne_twister(112358);

std::uniform_real_distribution<> uniform_float(0.0, 1.0);

float random_float() {
    return uniform_float(mersenne_twister);
}

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
        void calculate_sadows();    // Calculate windshadow
        void hopping();             // Select HOPPING_PRECENT of cells and execute hopping
        void calculate_gradients(); // Calculate gradients for each Cells
        void landslide();           // Execute landslides on Cells
        void fix_cells();           // 

        void simulate_step() {
            calculate_sadows();
            hopping();
            fix_cells();

            calculate_gradients();
            landslide();
            fix_cells();
        }

        void load_neigbours();
};

class Cell {
    private:
        int height;
        int temp_height;
        bool shadow;

        Cell* forward;
        Cell* backward;
        std::vector<Cell*> neigbours;
        std::vector<Cell*> diags;

    public:
        Cell(int height);

        // Height or temp height what is stacked or eroded?
        void erode() { height--; }
        void stack() { height++; }

        // Define a random float generator
        bool deposited() {
            if (shadow || random_float() <= DEPOSITE_CHANCE_LIGHT) {
                stack();
                return true;
            }
            else {
                return false;
            }
        }

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

int main () {
    return 0;
}